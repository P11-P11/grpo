import torch
import re
import logging
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

# Set up logging to display debug messages.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # 1. Define the custom reward function with debug logging.
    def chess_reward_func(prompts, completions, **kwargs):
        """
        Reward function for a chess reasoning task.
        Expected output format:
            <think> ... </think><box>move</box>

        Rewards:
        - +0.5 if the model correctly uses the <think> and <box> tokens.
        - +0.5 if the move inside <box> exactly matches the ground truth move.

        Arguments:
            prompts: List of prompts (chess board representations).
            completions: List of generated completions (strings or message dicts).
            kwargs: Additional dataset columns (expects 'expected_completion' as ground truth).

        Returns:
            List of floats (reward per sample): 0.0, 0.5, or 1.0.
        """
        logger.debug("chess_reward_func called!")
        logger.debug(f"Number of completions: {len(completions)}")

        ground_truths = kwargs.get("expected_completion")
        if ground_truths is None:
            logger.error("No 'expected_completion' found in kwargs!")
            raise ValueError("Expected ground truth column 'expected_completion' not found in kwargs")

        pattern = r"^<think>.*?</think><box>(.*?)</box>$"
        rewards = []

        for i, (gen, gt) in enumerate(zip(completions, ground_truths)):
            # If the generated output is in message-dict format, extract the 'content'
            if isinstance(gen, list) and isinstance(gen[0], dict):
                content = gen[0].get("content", "")
            else:
                content = gen

            logger.debug(f"Completion {i}: {content}")

            match = re.match(pattern, content)
            reward = 0.0
            if match:
                reward += 0.5  # Reward for using the correct tokens.
                move = match.group(1).strip()
                if move == gt:
                    reward += 0.5  # Additional reward for the correct move.
            else:
                logger.debug(f"No match for expected format in completion {i}.")

            logger.debug(f"Reward for sample {i}: {reward}")
            rewards.append(reward)

        return rewards
    
    def chess_reward_func(prompts, completions, **kwargs):
        """
        Updated reward function for a chess reasoning task with an easier initial incentive.
        Expected structure:
            <think> ... </think> <box> ... </box>
        
        Rewards:
        - +0.05 (one-time) if the output contains at least one of <think>, </think>, <box>, or </box>.
        - +0.5 if the output contains exactly one <think> ... </think> and one <box> ... </box>
            in the correct order and with no extra text after </box>.
        - +0.5 if the move extracted from <box> matches the ground truth.
                                                                                
        Penalties:
        - -0.5 if any of the tokens (<think>, </think>, <box>, </box>) appear more than once.
        - -0.5 if the <box> closure is not at the end of the text.
        
        Total reward per sample can be -0.5, 0.05, 0.5, or 1.05.
        """
        logger.debug("chess_reward_func called!")
        logger.debug(f"Number of completions: {len(completions)}")
        
        ground_truths = kwargs.get("expected_completion")
        if ground_truths is None:
            logger.error("No 'expected_completion' found in kwargs!")
            raise ValueError("Expected ground truth column 'expected_completion' not found in kwargs")
        
        rewards = []
        penalty = -0.5
        initial_bonus = 0.05
        initial_bonus_given = False

        for i, (gen, gt) in enumerate(zip(completions, ground_truths)):
            # Extract content if needed.
            if isinstance(gen, list) and isinstance(gen[0], dict):
                content = gen[0].get("content", "")
            else:
                content = gen

            logger.debug(f"Completion {i}: {content}")
            reward = 0.0
            
            # Count occurrences of tokens.
            count_think_open = content.count("<think>")
            count_think_close = content.count("</think>")
            count_box_open = content.count("<box>")
            count_box_close = content.count("</box>")
            
            # One-time initial bonus if any token is found.
            if not initial_bonus_given and (count_think_open or count_think_close or count_box_open or count_box_close):
                reward += initial_bonus
                initial_bonus_given = True
            
            # Check for exactly one occurrence each.
            if not (count_think_open == 1 and count_think_close == 1 and count_box_open == 1 and count_box_close == 1):
                logger.debug(f"Incorrect number of tokens in completion {i}.")
                reward += penalty
            else:
                # Find indices.
                think_open_index = content.find("<think>")
                think_close_index = content.find("</think>")
                box_open_index = content.find("<box>")
                box_close_index = content.find("</box>")
                
                # Check ordering: <think> ... </think> should come before <box> ... </box>
                if not (think_open_index < think_close_index < box_open_index < box_close_index):
                    logger.debug(f"Tokens are in the wrong order in completion {i}.")
                    reward += penalty
                else:
                    # Check that nothing follows after </box>
                    after_box = content[box_close_index + len("</box>"):].strip()
                    if after_box != "":
                        logger.debug(f"Extra text found after </box> in completion {i}.")
                        reward += penalty
                    else:
                        # Correct structure.
                        reward += 0.5

                        # Extract move from the <box> block.
                        move = content[box_open_index + len("<box>"):box_close_index].strip()
                        if move == gt:
                            reward += 0.5
                        else:
                            logger.debug(f"Move '{move}' does not match ground truth '{gt}' in completion {i}.")
            
            logger.debug(f"Reward for sample {i}: {reward}")
            rewards.append(reward)
        
        return rewards
    # 2. Prepare the datasets.
    csv_path = r'C:\Users\ezequ\OneDrive\Documentos\Facultad\Tesis\proyecto\low_train.csv'
    df = pd.read_csv(csv_path)
    df.columns = ['prompt', 'expected_completion']  # Rename columns for clarity
    # Add an instruction to each prompt
    instruction = (
        "Given the chess board below, find checkmate and explain your reasoning step-by-step.\n"
        "Your output should include a reasoning section enclosed by <think> and </think>, "
        "and the move enclosed by <box> and </box> tokens.\n"
    )
    df['prompt'] = instruction + df['prompt']

    # Create a small train and evaluation split.
    train_df = df.iloc[:10]
    eval_df = df.iloc[100:110]

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # 3. Load the model and tokenizer with 4-bit quantization.
    model_id = "Qwen/Qwen2-0.5B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    # Ensure a pad token is set (if not, use the EOS token).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Configure PEFT with LoRA (rank 128).
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=128,
        lora_alpha=32,
        target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # 5. Configure GRPOTrainer with extra logging.
    grpo_config = GRPOConfig(
        num_train_epochs=50,
        per_device_train_batch_size=12,
        learning_rate=1e-3, #6,
        logging_steps=1,          # Log every step.
        save_steps=500,
        max_completion_length=256,
        num_generations=6,
        log_completions=True,     # Log generated completions.
        output_dir="grpo_output",
        report_to= None
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=chess_reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=grpo_config,
    )

    # 6. Start training.
    trainer.train()

if __name__ == "__main__":
    main()

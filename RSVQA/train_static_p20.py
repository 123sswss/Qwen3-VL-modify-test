"""Pure LLM P20; no visual prompts or dynamic condition branch."""
from pathvqa.train_prompt_tuning import main

if __name__ == "__main__":
    raise SystemExit(main(dataset_name="rsvqa_lr"))

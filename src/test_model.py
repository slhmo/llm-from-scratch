import torch

from src.Configs import VOCAB_SIZE, N_LAYERS, TOKENIZER_PATH, MODEL_PATH
from src.Model.transformer import Transformer
from src.Tokenizer.BPE_tokenizer import BPETokenizer


def main():
    # tensorFloat32 core acceleration for faster matrix multiplications
    torch.set_float32_matmul_precision('high')
    tokenizer = BPETokenizer(VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)

    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)
    # Target the bound method directly and use max-autotune
    model.forward = torch.compile(model.forward, mode="default")
    model.load_weights(MODEL_PATH)

    # --- PREDICTION MODE ---
    prompt = tokenizer.encode("To be, or not to be")
    print("\n" + "=" * 50)
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        user_prompt = input("\nEnter your prompt: ")

        # Check if the user wants to exit
        if user_prompt.strip().lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break

        # Skip empty inputs
        if not user_prompt.strip():
            print("Prompt cannot be empty. Please try again.")
            continue

        # Encode, predict, and display
        prompt_tokens = tokenizer.encode(user_prompt)

        # You can adjust max_new_tokens or even make it dynamic if you want!
        generated_text = model.predict(prompt_tokens, max_new_tokens=500, tokenizer=tokenizer)

        print("\n--- Model Output ---")
        print(generated_text)
        print("-" * 20)



if __name__=='__main__':
    main()

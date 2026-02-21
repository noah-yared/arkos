from openai import OpenAI
import os
from mem0 import Memory


base_url = "http://0.0.0.0:30000/v1"

client = OpenAI(
    base_url=base_url,
    api_key="dummy",
)


os.environ["OPENAI_API_KEY"] = "sk"
config = {
    "vector_store": {
        "provider": "supabase",
        "config": {
            # "connection_string": "postgresql://user:password@host:port/database",
            "connection_string": "postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:54322/postgres",
            "collection_name": "memories",
            "index_method": "hnsw",  # Optional: defaults to "auto"
            "index_measure": "cosine_distance",  # Optional: defaults to "cosine_distance"
        },
    },
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qewn2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:30000/v1",
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"huggingface_base_url": "http://localhost:4444/v1"},
    },
}

memory = Memory.from_config(config)


def chat_with_memories(message: str, user_id: str = "root") -> str:
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct", messages=messages
    )
    assistant_response = response.choices[0].message.content

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response


def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()

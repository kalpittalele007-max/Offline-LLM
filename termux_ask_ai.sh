#!/data/data/com.termux/files/usr/bin/bash
set -eu
APP_DIR="$HOME/.local/share/offline-assistant"
MODEL_PATH="$APP_DIR/qwen3-1.7b-q4_k_m.gguf"
PART_PATH="$MODEL_PATH.part"
MODEL_URL="https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf?download=true"
PORT="${OFFLINE_ASSISTANT_PORT:-48736}"

if ! command -v pkg >/dev/null 2>&1; then
  echo "This script must be run in Termux."
  exit 1
fi

if ! command -v llama-server >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1 || ! command -v python >/dev/null 2>&1; then
  echo "Installing the local inference tools..."
  pkg install -y llama-cpp curl python
fi

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server was not found after installation. Update Termux packages and try again."
  exit 1
fi

mkdir -p "$APP_DIR"
if [ ! -s "$MODEL_PATH" ]; then
  echo "The language model will be downloaded once (about 1.3 GB). Internet is needed for this setup step."
  curl -L --fail --retry 3 --retry-delay 2 -C - -o "$PART_PATH" "$MODEL_URL"
  mv "$PART_PATH" "$MODEL_PATH"
fi

MODEL_BYTES=$(wc -c < "$MODEL_PATH" | tr -d ' ')
if [ "$MODEL_BYTES" -lt 800000000 ]; then
  echo "The model file looks incomplete: $MODEL_PATH"
  echo "Delete that file and run this script again while online to download it again."
  exit 1
fi

THREADS=$(nproc 2>/dev/null || echo 4)
llama-server --model "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" --ctx-size 4096 --threads "$THREADS" --parallel 1 >/dev/null 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export LOCAL_ASSISTANT_URL="http://127.0.0.1:$PORT"
export LOCAL_ASSISTANT_PID="$SERVER_PID"
python - "$@" <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = os.environ["LOCAL_ASSISTANT_URL"]
SERVER_PID = int(os.environ["LOCAL_ASSISTANT_PID"])
SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. Answer the user's question "
    "clearly and accurately. If you are uncertain, say so rather than making "
    "things up. You have no internet access and cannot know events after your "
    "training data unless the user provides that information."
)

def wait_until_ready():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        try:
            os.kill(SERVER_PID, 0)
        except OSError:
            raise RuntimeError("The local model server stopped while starting.")
        time.sleep(1)
    raise RuntimeError("The local model took too long to start. Try again after closing other apps.")

def main():
    try:
        wait_until_ready()
    except Exception as error:
        print("Could not start the local model: " + str(error))
        return 1

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\nOffline assistant ready. Type /help for commands.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not question:
            continue
        command = question.lower()
        if command in ("/exit", "/quit"):
            print("Goodbye.")
            return 0
        if command == "/help":
            print("Ask a question and press Enter. Commands: /reset clears this chat; /exit quits.\n")
            continue
        if command == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.\n")
            continue

        messages.append({"role": "user", "content": question})
        payload = json.dumps({
            "model": "local-model",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.8,
        }).encode("utf-8")
        request = urllib.request.Request(
            BASE_URL + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
            answer = data["choices"][0]["message"]["content"].strip()
            if not answer:
                raise ValueError("The local model returned an empty answer.")
            print("\nAssistant> " + answer + "\n")
            messages.append({"role": "assistant", "content": answer})
            if len(messages) > 17:
                messages = [messages[0]] + messages[-16:]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            print("\nLocal model error (" + str(error.code) + "): " + detail + "\n")
            messages.pop()
        except (urllib.error.URLError, TimeoutError) as error:
            print("\nCould not reach the local model: " + str(error) + "\n")
            messages.pop()
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            print("\nCould not read the local model response: " + str(error) + "\n")
            messages.pop()


if __name__ == "__main__":
    raise SystemExit(main())
PY

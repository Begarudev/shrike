"""Interactive terminal chat for the in-process garuda engine."""

from __future__ import annotations

import argparse
import readline
import sys
import time

from garuda.engine.engine import LLMEngine
from garuda.engine.request import SamplingParams


DEFAULT_MODEL = "models_cache/qwen2.5-0.5b-instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a local garuda model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--spec-ngram", type=int, default=0)
    return parser.parse_args()


def _ansi(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_help() -> None:
    print(
        """Commands:
  /help           Show this help
  /clear          Clear conversation history
  /metrics        Show engine metrics (shared history is mostly prefix-cache hits)
  /temp <float>   Set temperature; 0 uses greedy decoding
  /max <int>      Set maximum new tokens
  /spec on|off    Toggle prompt-lookup speculation
  /system <text>  Set or replace the system message
  /exit           Quit"""
    )


def _print_metrics(engine: LLMEngine) -> None:
    metrics = engine.metrics()
    width = max(len(key) for key in metrics)
    for key, value in metrics.items():
        print(f"{key:<{width}}  {value}")


def _stream_reply(
    engine: LLMEngine,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    use_color: bool,
) -> str | None:
    request_id: int | None = None
    generated: list[int] = []
    decode_buffer: list[int] = []
    reply_parts: list[str] = []
    started_at = time.perf_counter()
    first_token_at: float | None = None

    try:
        token_ids = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )["input_ids"]
        sampling = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        try:
            request_id = engine.add_request(token_ids, sampling, chat=False)
        except ValueError as error:
            print(f"[error] {error}")
            return None

        while engine.has_work:
            for output in engine.step():
                if output.req_id != request_id:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated.append(output.token_id)
                decode_buffer.append(output.token_id)
                text = engine.tokenizer.decode(
                    decode_buffer, skip_special_tokens=True
                )
                if not text.endswith("�"):
                    print(text, end="", flush=True)
                    reply_parts.append(text)
                    decode_buffer.clear()
    except KeyboardInterrupt:
        if request_id is not None:
            request = engine.requests.get(request_id)
            if request is not None:
                try:
                    engine.scheduler.finish(request, "aborted")
                except ValueError:
                    if request in engine.scheduler.waiting:
                        engine.scheduler.waiting.remove(request)
                    engine.block_manager.free(request)
                del engine.requests[request_id]
        print("\n[aborted]")
        return None

    elapsed = time.perf_counter() - started_at
    ttft_ms = (
        (first_token_at - started_at) * 1000 if first_token_at is not None else 0.0
    )
    rate = len(generated) / max(elapsed, 1e-9)
    print()
    status = (
        f"⏱ {len(generated)} tok · {elapsed:.1f}s · {rate:.0f} tok/s "
        f"· TTFT {ttft_ms:.0f}ms"
    )
    print(_ansi(status, "2", use_color))
    return "".join(reply_parts)


def main() -> None:
    args = _parse_args()
    readline.set_auto_history(True)
    use_color = sys.stdout.isatty()

    print(f"Loading {args.model} ...", flush=True)
    load_started = time.perf_counter()
    engine = LLMEngine(args.model, spec_ngram=args.spec_ngram)
    load_time = time.perf_counter() - load_started
    print(
        f"garuda ready · {args.model} · {engine.block_manager.num_blocks} KV blocks "
        f"· loaded in {load_time:.1f}s"
    )

    messages: list[dict[str, str]] = []
    temperature = 0.7
    max_new_tokens = 512
    prompt = (
        "\001\033[1;36m\002garuda> \001\033[0m\002"
        if use_color
        else "garuda> "
    )

    while True:
        try:
            line = input(prompt)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nUse /exit to quit.")
            continue

        if not line.strip():
            continue

        if line.startswith("/"):
            command, _, argument = line.strip().partition(" ")
            argument = argument.strip()

            if command == "/exit":
                break
            if command == "/help":
                _print_help()
            elif command == "/clear":
                messages.clear()
                print("Conversation cleared.")
            elif command == "/metrics":
                _print_metrics(engine)
            elif command == "/temp":
                try:
                    value = float(argument)
                    if not 0 <= value < float("inf"):
                        raise ValueError
                except ValueError:
                    print("Usage: /temp <non-negative float>")
                else:
                    temperature = value
                    print(f"Temperature set to {temperature:g}.")
            elif command == "/max":
                try:
                    value = int(argument)
                    if value <= 0:
                        raise ValueError
                except ValueError:
                    print("Usage: /max <positive int>")
                else:
                    max_new_tokens = value
                    print(f"Maximum new tokens set to {max_new_tokens}.")
            elif command == "/spec":
                if argument == "on":
                    engine.scheduler.spec_ngram = 2
                    print("Speculation on; takes effect at /temp 0.")
                elif argument == "off":
                    engine.scheduler.spec_ngram = 0
                    print("Speculation off.")
                else:
                    print("Usage: /spec on|off")
            elif command == "/system":
                if not argument:
                    print("Usage: /system <text>")
                elif messages and messages[0]["role"] == "system":
                    messages[0]["content"] = argument
                    print("System message updated.")
                else:
                    messages.insert(0, {"role": "system", "content": argument})
                    print("System message set.")
            else:
                print(f"Unknown command: {command}. Type /help for commands.")
            continue

        messages.append({"role": "user", "content": line})
        reply = _stream_reply(
            engine, messages, temperature, max_new_tokens, use_color
        )
        if reply is not None:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

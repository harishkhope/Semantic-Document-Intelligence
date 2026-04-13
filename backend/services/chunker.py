MIN_CHUNK_LENGTH = 50


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text_content = text[start:end]

        if len(chunk_text_content) >= MIN_CHUNK_LENGTH:
            chunks.append(
                {
                    "text": chunk_text_content,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                }
            )
            chunk_index += 1

        if end == text_length:
            break

        start += chunk_size - chunk_overlap

    return chunks

"""End-to-end Route-A Add/Search orchestration."""
from __future__ import annotations

import hashlib
from .source_time import source_created_at
from typing import Any

from .config import Settings
from .captioning import CAPTION_MODE_COMPLEMENT, source_captions_for_messages
from .media import (
    MAX_ADD_IMAGE_BYTES,
    MAX_SEARCH_IMAGE_BYTES,
    image_urls,
    plain_text,
    response_image_bytes,
    validate_parts,
)
from .providers import EmbeddingProvider, VisionLanguageProvider
from .retrieval import hybrid_rank
from .schemas import (
    AddRequest,
    AddResponse,
    ContentPart,
    ImagePart,
    SearchItem,
    SearchRequest,
    SearchResponse,
    TextPart,
)
from .store import SQLiteStore, payload_digest


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _chunk_text(text: str, max_characters: int = 6000) -> list[str]:
    """Conservatively stay below text-embedding-v4's 8,192-token row limit.

    Character chunks are intentionally conservative for CJK. This is partitioning rather than
    truncation: every character remains represented by at least one embedding view.
    """
    text = text.strip()
    if not text:
        return []
    output: list[str] = []
    while len(text) > max_characters:
        boundary = max(
            text.rfind("\n", 0, max_characters),
            text.rfind(". ", 0, max_characters),
            text.rfind("。", 0, max_characters),
            text.rfind(" ", 0, max_characters),
        )
        if boundary < max_characters // 2:
            boundary = max_characters
        else:
            boundary += 1
        output.append(text[:boundary].strip())
        text = text[boundary:].strip()
    if text:
        output.append(text)
    return output


def _flatten_messages(request: AddRequest) -> tuple[list[dict], str, list[str]]:
    """Make one stable AML candidate while preserving message and part order."""
    output: list[dict] = []
    transcript: list[str] = []
    images: list[str] = []
    for message in request.messages:
        first = True
        transcript.append(f"[{message.role}]")
        for part in message.content:
            if isinstance(part, TextPart):
                text = f"[{message.role}] {part.text}" if first else part.text
                output.append({"type": "text", "text": text})
                transcript.append(part.text)
            else:
                if first:
                    output.append({"type": "text", "text": f"[{message.role}]"})
                output.append(part.model_dump())
                images.append(part.image_url.url)
                transcript.append("[image]")
            first = False
    return output, "\n".join(transcript).strip(), images


def _descriptor_views(
    rows: list[dict], source_captions: list[str] | None = None, *, combine_captions: bool = False
) -> list[tuple[str, str]]:
    fields = (
        "scene_summary", "objects", "people", "actions", "attributes", "ocr_text",
        "spatial_relations", "time_place_cues", "fine_details", "search_keywords",
    )
    views: list[tuple[str, str]] = []
    for fallback_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            image_index = int(row.get("image_index", fallback_index))
        except (TypeError, ValueError):
            image_index = fallback_index
        joint_parts: list[str] = []
        for field in fields:
            text = _as_text(row.get(field))
            if text:
                views.append((f"visual.{field}", f"Image {image_index}: {text}"))
                joint_parts.append(f"{field.replace('_', ' ')}: {text}")
        if joint_parts:
            views.append(("visual.joint", f"Image {image_index}. " + " | ".join(joint_parts)))
        if combine_captions:
            source_caption = ""
            if source_captions is not None and 0 <= image_index < len(source_captions):
                source_caption = source_captions[image_index].strip()
            combined_parts = []
            if source_caption:
                combined_parts.append(f"Source caption: {source_caption}")
            if joint_parts:
                combined_parts.append("Generated complementary caption: " + " | ".join(joint_parts))
            if combined_parts:
                views.append((
                    "visual.caption_combined",
                    f"Image {image_index}. " + " | ".join(combined_parts),
                ))
    return views


class RouteAService:
    def __init__(self, settings: Settings, *, store: SQLiteStore | None = None,
                 embedder: EmbeddingProvider | None = None,
                 vlm: VisionLanguageProvider | None = None):
        self.settings = settings
        self.store = store or SQLiteStore(
            settings.storage_path, settings.add_processing_timeout_seconds
        )
        self.embedder = embedder or EmbeddingProvider(settings)
        self.vlm = vlm or VisionLanguageProvider(settings)

    def add(self, request: AddRequest) -> AddResponse:
        payload = request.model_dump(mode="json")
        digest = payload_digest(payload)
        cached = self.store.claim_add(request.request_id, digest)
        if cached is not None:
            return AddResponse.model_validate(cached)
        try:
            total_image_bytes = 0
            for message in request.messages:
                total_image_bytes += validate_parts(message.content, total_limit=MAX_ADD_IMAGE_BYTES)
                if total_image_bytes > MAX_ADD_IMAGE_BYTES:
                    from .errors import BadRequest
                    raise BadRequest("decoded images exceed the 30 MiB Add request limit")

            content, transcript, images = _flatten_messages(request)
            source_captions = source_captions_for_messages(request.messages)
            descriptors = self.vlm.describe_images(
                transcript, images, source_captions=source_captions
            )

            raw_views: list[tuple[str, str]] = [("conversation", transcript)]
            for index, message in enumerate(request.messages):
                text = plain_text(message.content)
                if text:
                    raw_views.append((f"message.{message.role}", f"[{message.role} {index}] {text}"))
            visual_views = _descriptor_views(
                descriptors,
                source_captions,
                combine_captions=(
                    self.settings.caption_generation_mode == CAPTION_MODE_COMPLEMENT
                ),
            )
            raw_views.extend(visual_views)
            if visual_views:
                joined_visual = "\n".join(text for _, text in visual_views if text)
                raw_views.append(("conversation_visual_joint", f"{transcript}\n{joined_visual}"))

            # Remove exact duplicates but keep semantically distinct views as separate retrieval lanes.
            deduped: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for kind, text in raw_views:
                chunks = _chunk_text(text)
                for chunk_index, chunk in enumerate(chunks):
                    chunk_kind = kind if len(chunks) == 1 else f"{kind}.part{chunk_index}"
                    key = (chunk_kind, chunk)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(key)
            vectors = self.embedder.encode([text for _, text in deduped])

            memory_id = _stable_id("mem", request.user_id, request.request_id)
            timestamps = [message.timestamp for message in request.messages
                          if source_created_at(message.timestamp) is not None]
            source_timestamp_ms = min(timestamps) if timestamps else None
            # Empty sentinel preserves the existing NOT NULL schema, never ingestion time.
            created_at = source_created_at(source_timestamp_ms) or ""
            search_text = "\n".join(text for _, text in deduped)
            memory = {
                "memory_id": memory_id,
                "request_id": request.request_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "source_timestamp_ms": source_timestamp_ms,
                "content": content,
                "search_text": search_text,
                "created_at": created_at,
            }
            views = [
                {
                    "view_id": _stable_id("view", memory_id, str(index), kind, text),
                    "kind": kind,
                    "text": text,
                    "embedding": vectors[index],
                }
                for index, (kind, text) in enumerate(deduped)
            ]
            response = AddResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            # This transaction is the synchronous boundary: both raw evidence and every index
            # vector are durable before the success response is returned.
            self.store.complete_add(memory, views, response.model_dump(mode="json"))
            return response
        except Exception:
            self.store.abandon_add(request.request_id, digest)
            raise

    def search(self, request: SearchRequest, *, _seed_ranked: list[tuple[str,float]] | None = None, trace: dict[str, Any] | None = None) -> SearchResponse:
        """Return post-rerank evidence; optionally collect local-only retrieval diagnostics.

        ``trace`` is never part of the AML HTTP response. It exists so offline benchmark runners can
        compare coarse retrieval, multimodal reranking, neighbor expansion, and the final response
        without weakening the fixed Search API schema.
        """
        if trace is not None:
            trace.clear()
            trace.update({
                "trace_version": 2,
                "search_output_stage": "post_rerank_neighbor_assembled",
                "limits": {
                    "requested_top_k": request.top_k,
                    "retrieve_pool": self.settings.retrieve_pool,
                    "rerank_top_n": self.settings.rerank_top_n,
                    "return_limit": self.settings.return_limit,
                    "comparison_cutoff": min(request.top_k, self.settings.return_limit),
                },
            })
        validate_parts(request.query, total_limit=MAX_SEARCH_IMAGE_BYTES)
        if _seed_ranked is None:
            views = self.store.views_for_user(request.user_id)
            if not views:
                if trace is not None:
                    trace.update({
                        "query_plan": {}, "query_texts": [], "query_image_count": 0,
                        "coarse_candidates": [], "reranked_candidates": [],
                        "returned_candidates": [],
                        "summary": {
                            "coarse_candidate_count": 0, "rerank_attempted_count": 0,
                            "rerank_scored_count": 0, "returned_count": 0,
                        },
                    })
                return SearchResponse(data=[])
            question = plain_text(request.query)
            query_images = image_urls(request.query)
            options = request.options or []
            plan = self.vlm.plan_query(question, options, query_images)
    
            query_texts = [question]
            visual_description = _as_text(plan.get("visual_description"))
            intent = _as_text(plan.get("intent"))
            anchors = plan.get("text_anchors", [])
            keywords = _as_text(plan.get("keywords"))
            if intent:
                query_texts.append(intent)
            if visual_description:
                query_texts.append(visual_description)
            if isinstance(anchors, list):
                query_texts.extend(str(anchor).strip() for anchor in anchors if str(anchor).strip())
            if keywords:
                query_texts.append(keywords)
            if options:
                query_texts.append("Possible answer concepts: " + " | ".join(options))
            query_texts = list(dict.fromkeys(
                chunk for text in query_texts if text for chunk in _chunk_text(text)
            ))
            if not query_texts:
                query_texts = ["visual query"]
    
            if trace is not None:
                trace.update({
                    "query_plan": plan,
                    "query_texts": query_texts,
                    "query_image_count": len(query_images),
                    "fusion_weights": {"coarse": 0.45, "reranker": 0.55},
                    "neighbor_window": self.settings.neighbor_window,
                    "assembly_strategy": (
                        "base_first_neighbor_quota"
                        if self.settings.base_first_quota > 0 else "block_oriented"
                    ),
                    "base_first_quota": self.settings.base_first_quota,
                    "lexical_retriever": "bm25" if self.settings.use_bm25 else "token_overlap",
                    "bm25": (
                        {"k1": self.settings.bm25_k1, "b": self.settings.bm25_b}
                        if self.settings.use_bm25 else None
                    ),
                })
    
            query_vectors = self.embedder.encode(query_texts)
            combined_query = "\n".join(query_texts)
            ranked = hybrid_rank(
                query_vectors,
                combined_query,
                views,
                self.settings.retrieve_pool,
                use_bm25=self.settings.use_bm25,
                bm25_k1=self.settings.bm25_k1,
                bm25_b=self.settings.bm25_b,
            )
        else:
            ranked = list(_seed_ranked)
            combined_query = ''
            query_images = []
        memory_ids = [memory_id for memory_id, _ in ranked]
        memories = self.store.memories_by_ids(request.user_id, memory_ids)
        coarse_rank = {memory_id: index for index, memory_id in enumerate(memory_ids, 1)}
        fused_scores = dict(ranked)
        max_retrieval = max(fused_scores.values(), default=1.0)
        if trace is not None:
            trace["coarse_candidates"] = [
                {
                    "memory_id": memory_id,
                    "coarse_rank": coarse_rank[memory_id],
                    "coarse_score": float(score),
                    "coarse_score_normalized": float(score / max_retrieval),
                }
                for memory_id, score in ranked
            ]

        # Stage 2 is the multimodal reranker: it sees original query/candidate pixels and corrects
        # caption-only misses. It scores evidence relevance and never produces the final answer.
        rerank_ids = memory_ids[:self.settings.rerank_top_n]
        rerank_input = []
        for memory_id in rerank_ids:
            memory = memories.get(memory_id)
            if memory is None:
                continue
            parts = [TextPart.model_validate(part) if part.get("type") == "text"
                     else ImagePart.model_validate(part) for part in memory["content"]]
            rerank_input.append({
                "id": memory_id,
                "text": memory["search_text"],
                # Limit visual payload while retaining direct pixel comparison for the strongest candidates.
                "images": image_urls(parts)[:2],
            })
        rerank_scores: dict[str, float] = {}
        # Small batches avoid constructing a single request containing dozens of original images.
        for start in range(0, len(rerank_input), 4):
            rerank_scores.update(
                self.vlm.rerank(combined_query, query_images, rerank_input[start:start + 4])
            )

        final_score: dict[str, float] = {}
        for position, memory_id in enumerate(memory_ids):
            retrieval = fused_scores[memory_id] / max_retrieval
            if memory_id in rerank_scores:
                final_score[memory_id] = 0.45 * retrieval + 0.55 * (rerank_scores[memory_id] / 100.0)
            else:
                # Keep non-reranked tail below a comparably relevant reranked item without deleting it.
                final_score[memory_id] = 0.45 * retrieval - position * 1e-9
        base_order = sorted(memory_ids, key=lambda item: (-final_score[item], item))
        reranked_rank = {memory_id: index for index, memory_id in enumerate(base_order, 1)}
        if trace is not None:
            attempted_ids = {row["id"] for row in rerank_input}
            trace["reranked_candidates"] = [
                {
                    "memory_id": memory_id,
                    "coarse_rank": coarse_rank[memory_id],
                    "reranked_rank": reranked_rank[memory_id],
                    "rank_change": coarse_rank[memory_id] - reranked_rank[memory_id],
                    "coarse_score": float(fused_scores[memory_id]),
                    "coarse_score_normalized": float(fused_scores[memory_id] / max_retrieval),
                    "rerank_status": (
                        "scored" if memory_id in rerank_scores else
                        "missing" if memory_id in attempted_ids else
                        "not_attempted"
                    ),
                    "reranker_score": (
                        float(rerank_scores[memory_id]) if memory_id in rerank_scores else None
                    ),
                    "fused_score": float(final_score[memory_id]),
                }
                for memory_id in base_order
            ]

        limit = min(request.top_k, self.settings.return_limit)
        assembled: list[tuple[dict, float, str, str | None]] = []
        seen: set[str] = set()
        base_first_quota = min(self.settings.base_first_quota, limit)
        if base_first_quota > 0:
            # Reserve the leading Q positions for ranked base evidence, then spend only the
            # remaining shared budget on their session neighbors. If there are too few unique
            # neighbors, later ranked bases backfill the unused slots.
            selected_bases: list[str] = []
            for memory_id in base_order:
                memory = memories.get(memory_id)
                if memory is None or memory_id in seen:
                    continue
                seen.add(memory_id)
                selected_bases.append(memory_id)
                assembled.append((memory, final_score[memory_id], "reranked_base", None))
                if len(selected_bases) >= base_first_quota:
                    break

            for memory_id in selected_bases:
                if len(assembled) >= limit:
                    break
                for neighbor in self.store.session_neighbors(
                    request.user_id, memory_id, self.settings.neighbor_window
                ):
                    if len(assembled) >= limit:
                        break
                    if neighbor["memory_id"] not in seen:
                        seen.add(neighbor["memory_id"])
                        assembled.append((
                            neighbor, final_score[memory_id] * 0.97,
                            "session_neighbor", memory_id,
                        ))

            if len(assembled) < limit:
                for memory_id in base_order:
                    if len(assembled) >= limit:
                        break
                    memory = memories.get(memory_id)
                    if memory is None or memory_id in seen:
                        continue
                    seen.add(memory_id)
                    assembled.append((memory, final_score[memory_id], "reranked_base", None))
        else:
            for memory_id in base_order:
                if len(assembled) >= limit:
                    break
                memory = memories.get(memory_id)
                if memory is None or memory_id in seen:
                    continue
                seen.add(memory_id)
                assembled.append((memory, final_score[memory_id], "reranked_base", None))
                for neighbor in self.store.session_neighbors(
                    request.user_id, memory_id, self.settings.neighbor_window
                ):
                    if len(assembled) >= limit:
                        break
                    if neighbor["memory_id"] not in seen:
                        seen.add(neighbor["memory_id"])
                        # A small penalty preserves the base hit before its conversational context.
                        assembled.append((
                            neighbor, final_score[memory_id] * 0.97,
                            "session_neighbor", memory_id,
                        ))

        results: list[SearchItem] = []
        returned_trace: list[dict[str, Any]] = []
        media_skipped: list[str] = []
        response_media_bytes = 0
        previous_reported_score = float("inf")
        for memory, score, source, expanded_from in assembled:
            parts: list[ContentPart] = [
                TextPart.model_validate(part) if part.get("type") == "text"
                else ImagePart.model_validate(part)
                for part in memory["content"]
            ]
            candidate_bytes = response_image_bytes(parts)
            if response_media_bytes + candidate_bytes > MAX_SEARCH_IMAGE_BYTES:
                media_skipped.append(memory["memory_id"])
                continue
            response_media_bytes += candidate_bytes
            # Make the public score strictly monotone with the actual assembled rank, as required
            # by the contract. This also covers base-first, where a later neighbor may inherit a
            # score higher than an earlier low-ranked base.
            reported_score = min(float(score), previous_reported_score - 1e-6)
            previous_reported_score = reported_score
            results.append(SearchItem(
                id=memory["memory_id"], content=parts, score=reported_score,
                created_at=source_created_at(memory.get("source_timestamp_ms")),
            ))
            returned_trace.append({
                "memory_id": memory["memory_id"],
                "returned_rank": len(results),
                "source": source,
                "expanded_from_memory_id": expanded_from,
                "pre_monotone_score": float(score),
                "reported_score": float(reported_score),
            })
        if trace is not None:
            trace["returned_candidates"] = returned_trace[:request.top_k]
            trace["media_budget_skipped_memory_ids"] = media_skipped
            trace["summary"] = {
                "coarse_candidate_count": len(memory_ids),
                "rerank_requested_count": len(rerank_ids),
                "rerank_attempted_count": len(rerank_input),
                "rerank_scored_count": len(rerank_scores),
                "rerank_missing_count": len({row["id"] for row in rerank_input} - set(rerank_scores)),
                "assembly_strategy": (
                    "base_first_neighbor_quota"
                    if self.settings.base_first_quota > 0 else "block_oriented"
                ),
                "base_first_quota": self.settings.base_first_quota,
                "returned_count": min(len(results), request.top_k),
                "media_budget_skipped_count": len(media_skipped),
            }
        return SearchResponse(data=results[:request.top_k])

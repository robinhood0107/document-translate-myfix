# Gemma result cache and exact translation memory

Comic Translate provides two separate local fast paths for managed Gemma translation.

## Persistent block-result cache

The result cache reuses sanitized raw translations only when the complete output identity matches. The identity covers ordered full-group context, target index/key, languages, extra context, grouped mode and size, prompt/profile/schema, every sampler value, model SHA-256, runtime fingerprint, dictionary hash, and sanitizer/guard/TM versions.

A complete hit is resolved before runtime readiness and therefore does not start a stopped Gemma container. A partial hit keeps every original block marker in context and requests only unresolved `requested_blocks`. Current translation-result dictionary substitutions are applied exactly once after both hits and misses.

## Approved exact translation memory

Exact translation memory is separate from the result cache and the post-translation substitution dictionary. New translations are collected only as unapproved candidates. They cannot bypass Gemma until the user explicitly approves the source-to-translation pair.

Matching normalizes Unicode NFC, line endings, and outer whitespace only. It does not use fuzzy matching, embeddings, or semantic search. Conflicting approved translations for the same source/language pair are treated as ambiguous misses.

## Settings and local data

Open **Settings → User Dictionaries → Exact Translation Memory** to:

- enable or disable persistent result caching;
- enable exact TM and unapproved-candidate collection;
- configure result-cache and candidate retention limits;
- approve, unapprove, or delete TM entries;
- clear only the result cache;
- import or export exact TM JSON.

The SQLite database contains sensitive source and translation text in the application user-data directory. It stores no raw images. Imported approved entries can bypass Gemma, so the app asks for confirmation before trusting them.

If SQLite is locked, corrupt, or has an incompatible schema, caching is disabled for that run and normal translation continues. The database is not automatically deleted or rewritten, and failed management operations are reported instead of being presented as successful.

Approved entries are never removed by bounded candidate retention. Result-cache entries and unapproved candidates are pruned by least-recent use when their configured limits are exceeded.

llama.cpp `cache_prompt`/`cache-ram` reuse prompt prefill only; they are distinct from these persistent translation-result fast paths.

# HN Digest: Self-hosting Postgres

## TL;DR
Self-hosting Postgres is commonly managed with Kubernetes operators like `cloudnative-pg` for automated failover and management [#38844923]. However, this approach is debated, with some arguing Kubernetes adds unnecessary complexity and is not well-suited for stateful systems like Postgres [#43588074].

## Consensus Views
The community largely agrees that a mature ecosystem exists for self-hosting Postgres, offering solutions for various scales and operational philosophies. Some prefer lightweight setups, noting an application "does not require Docker" [#40523806], while others advocate for large-scale, orchestrated systems.

## Main Arguments For
- Kubernetes provides a powerful base for Postgres with a well‑tested failover system [#38844923].
- Cohesive autonomic systems improve fault‑tolerance and consistency [#38844923].
- Operators like `cloudnative-pg` can be used to manage Postgres deployments, automating setup and maintenance [#38843996].
- Sandboxing solutions are essential for self-hosted infrastructure to provide solid security guarantees [#46455222].
- Self-hosted server software should be written in languages like Rust, Go, or C++ for performance and reliability [#40530099].

## Main Arguments Against
- For some applications, self-hosting a separate database is unnecessary complexity, as the app "does not require a separate database" or a Redis instance [#40523806].
- Kubernetes is often criticized as being "not well suited for stateful systems like Postgres that need resource management" [#43588074].
- The overhead of a full orchestration platform like Kubernetes may be excessive for smaller deployments.

## Where the Community Disagrees
**Kubernetes for Postgres Management**
- Side A: Kubernetes provides a powerful base for Postgres with a well‑tested failover system, and cohesive autonomic systems improve fault-tolerance and consistency [#38844923].
- Side B: Kubernetes is not well suited for stateful systems like Postgres that need resource management [#43588074].

## Alternatives Mentioned
| Tool | Times | Context |
|------|-------|---------|
| Kubernetes | 7 | Provides a powerful base for Postgres with a well‑tested failover system. |
| Docker | 7 | The app does not require Docker; images are based on Debian. |
| PgDog | 5 | A proxy/tool that currently does not manage DDL replication for logical replicas. |
| Hatchet | 8 | A task queue that supports webhooks for each task. |
| OVH | 2 | An inexpensive cloud service provider for self-hosted infrastructure. |

## Notable War Stories (firsthand)
- [#38843996] "setup cloudnative-pg operator to manage Postgres"
- [#43575497] "Many tasks in backlog, many workers, and simultaneous long‑polling can cause high CPU spikes and database deterioration."
- [#40531370] "The VPS image is 200 MB in size."
- [#43586136] "The queue table fits in RAM with a 100% buffer hit rate."
- [#38844947] "Kubernetes makes sense at large scale, handling hundreds of ES nodes across many clusters."

## Evidence Quality Note
The evidence is mostly anecdotal and opinion-based, consisting of firsthand experiences with tools like `cloudnative-pg` and architectural critiques of platforms like `Kubernetes` [#38843996, #43588074]. While some claims cite specific tool features, there is a lack of broad, objective benchmarks comparing different self-hosting strategies. Confidence in the claims should be moderate, as they reflect individual experiences rather than large-scale studies.
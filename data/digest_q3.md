# HN Digest: SQLite in production

## TL;DR
SQLite is a viable embedded database for production, especially with WAL mode and robust backup strategies like the native `.backup` command or tools such as Litestream. However, it requires careful handling of concurrency and conflict resolution, and its feature set may be limiting compared to client-server databases. [#47641494,#39956255]

## Consensus Views
- SQLite is a viable embedded database, especially when used with WAL mode for better concurrency.
- Using the `.backup` command or a dedicated tool like Litestream is considered essential for data safety in production.
- The community largely agrees that SQLite's simplicity is a major advantage for specific use cases, but it is not a drop-in replacement for all production database needs.
- PRAGMA journal_mode = WAL is the standard configuration for enabling concurrent reads.

## Main Arguments For
- SQLite provides a native `.backup` command that is safer for database backups than using a simple `cp` command. [#47641494]
- Using WAL mode (`PRAGMA journal_mode = WAL`) enables concurrent reads and serialized writes, which is crucial for performance in production. [#36584827,#39957351]
- SQLite is used successfully in production by businesses of significant scale, often in combination with tools like Litestream for continuous backups. [#39956255]
- It is a self-contained, serverless database with no external dependencies, simplifying deployment and operations.
- Backend developers must be prepared to handle conflict resolution to avoid data loss in distributed systems using SQLite. [#38481961]

## Main Arguments Against
- SQLite can suffer from "database locked" errors during high-concurrency operations, even in WAL mode. [#39835880]
- A specific war story highlighted that frequent deployments (11 pushes in 2 hours) with concurrent SQLite access led to overlapping deploys and potential issues. [#47638212]
- SQLite lacks certain advanced features like native timestamp support, which can limit functionality for complex synchronization tasks compared to Postgres. [#38480612]
- The responsibility for managing data consistency and conflict resolution falls entirely on the developer, adding complexity to the application logic. [#38481961]

## Where the Community Disagrees
**Concurrency and Data Safety**
- Side A: SQLite is safe for production concurrency when configured correctly. "WAL mode enables concurrent reads and serialized writes via WAL mode." "Using the .backup command reduces risk of data loss/corruption compared to using cp." "Use SQLite for production services in an 8-figure ARR business with Litestream backups." [#39957351,#47641494,#39956255]
- Side B: SQLite has inherent concurrency limitations that can lead to data access errors. "WAL mode does not prevent 'database locked' errors; they can still occur." "11 pushes in 2 h caused overlapping Kamal deploys with concurrent SQLite access." [#39835880,#47638212]

## Alternatives Mentioned
| Tool | Times | Context |
|------|-------|---------|
| Litestream | 8 | Backups for SQLite in production services. |
| PowerSync | 5 | Server-authoritative sync with local SQLite. |
| Postgres | 2 | A more feature-rich alternative, especially for sync. |
| Kamal | 4 | A deployment tool that can cause concurrency issues with SQLite. |
| Docker | 3 | WAL mode is confirmed to work across Docker boundaries. |

## Notable War Stories (firsthand)
- [#47638212] "11 pushes in 2 h caused overlapping Kamal deploys with concurrent SQLite access"
- [#38481961] "Backend developers must handle conflicts to avoid data loss"
- [#39835880] "WAL mode does not prevent 'database locked' errors; they can still occur"
- [#36584827] "PRAGMA journal_mode = WAL"
- [#39957351] "Enables concurrent reads and serialized writes via WAL mode"

## Evidence Quality Note
The evidence is primarily anecdotal, consisting of developer war stories and opinions on specific use cases. While specific tools like Litestream and benchmarks like the Kamal deploy issue are mentioned, the consensus is built on practical experience rather than formal, comparative studies. Readers should weigh these firsthand accounts heavily when making a decision.
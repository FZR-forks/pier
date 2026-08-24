You must delegate this work rather than doing it yourself.

Spawn exactly two sub-agents with the spawn_agent tool, one per file, and wait for both:

1. One sub-agent writes /app/first.txt containing exactly: FIRST
2. One sub-agent writes /app/second.txt containing exactly: SECOND

Do not create either file yourself. After both sub-agents report completion, write
/app/report.txt containing exactly: DELEGATED

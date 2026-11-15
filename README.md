# entry-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/roy-ah/entry-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/roy-ah/entry-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One Node 22 server that serves the OpenAI and Anthropic completion APIs behind a single endpoint, answering every prompt with a real completion from the best-rated free OpenRouter model that meets it, with failover across ranked candidates when a provider refuses.

## Run it

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://localhost:8080 OPENAI_BASE_URL=http://localhost:8080/v1 node main.ts
```

Print the catalog it registers, without starting a server:

```sh
MAKEFUN_PROMPT_FILE=prompt.txt node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=p2wx7twh&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>

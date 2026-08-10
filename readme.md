# 启动7B
docker compose --profile 7b up -d

# 启动8B
docker compose --profile 8b up -d

# 启动32B
docker compose --profile 32b up -d

# 停止对应服务
docker compose --profile 7b down

# 查看日志
docker compose --profile 7b logs -f qwen2.5vl-7b-fp8
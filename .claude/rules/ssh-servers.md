# Rule: SSH Server Access — Always Available

## GPU Server (primary for ANIMA work)
```bash
ssh datai_srv7_development
```
- IP: 34.124.148.215 | User: datai | Key: ~/.ssh/datai_srv
- 8x NVIDIA L4 (23GB each)
- Modules: /mnt/forge-data/modules/ (93 ANIMA modules)
- Models: /mnt/forge-data/models/
- Datasets: /mnt/forge-data/datasets/
- Docker images: anima-base:jazzy, anima-serve:jazzy
- HF token: cached (~/.cache/huggingface/token → ilessio-aiflowlab)
- Skills/rules symlinked from /home/datai/.claude/
- tmux for sessions, nohup+disown for training

## When to SSH
- Training modules → `ssh datai_srv7_development`
- Checking tmux sessions → `ssh datai_srv7_development "tmux list-sessions"`
- Data downloads → start on server, not Mac
- Docker builds → on server (has NVIDIA runtime)

## Full server list
See `~/.claude/ssh_servers.md` for all AIFLOW LABS servers.

## DO NOT
- Never store SSH keys in git or memory files
- Never hardcode IPs in code — use Host aliases from ~/.ssh/config
- Never run training on Mac — always on GPU server

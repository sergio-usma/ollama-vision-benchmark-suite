# OllamaVision Benchmark Suite — Documentation Index

Welcome to the full documentation for OllamaVision Benchmark Suite v4.0.

---

## Documents

| Document | Contents |
|----------|----------|
| [Architecture](architecture.md) | MVC design, module descriptions, key design decisions |
| [Configuration](configuration.md) | Full `config.py` reference — all tunable parameters |
| [Scoring System](scoring-system.md) | Efficiency score formula, recommendation labels, tuning |
| [Dashboard Guide](dashboard-guide.md) | Tab-by-tab guide to reading the HTML dashboard |
| [Hardware Monitoring](hardware-monitoring.md) | jtop integration, simulation mode, power architecture |

---

## Quick Navigation

**I want to...**

- **Run my first benchmark** → See [README.md](../README.md#-quick-start)
- **Understand what the score means** → See [Scoring System](scoring-system.md)
- **Change category thresholds** → See [Configuration](configuration.md#minimum-tps-per-category)
- **Add a new model** → See [Configuration](configuration.md#model-category-override)
- **Read the dashboard** → See [Dashboard Guide](dashboard-guide.md)
- **Understand the architecture** → See [Architecture](architecture.md)
- **Configure hardware monitoring** → See [Hardware Monitoring](hardware-monitoring.md)
- **Troubleshoot** → See [README.md](../README.md#-troubleshooting)

---

## Version

- **Suite version:** 4.0
- **Target hardware:** NVIDIA Jetson AGX Orin (64 GB unified memory)
- **Compatible:** Any Linux with Ollama (use `--simulate` without Jetson)
- **Python:** 3.10+
- **Ollama:** 0.1.30+

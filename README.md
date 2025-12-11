This Chess environment is based on https://github.com/patrik-ha/explainable-minichess

## Microchess Agents 
This project explores intelligent gameplay on a compact 5×4 Microchess board, where classic chess dynamics shrink into a tighter, trickier arena. I implemented multiple agents using minimax search, heuristic evaluation, and (optionally) learning-based strategies. The environment, agents, and autograding pipeline were provided as part of the assignment framework.

### Project Structure
```
agents/
 ├── task1_agent.py      # Depth-2 minimax vs Random agent
 ├── task2_agent.py      # Depth-4 minimax vs Rational agent
 ├── task3_agent.py      # Best agent (search + heuristics/RL)
 ├── base_agent.py
extras/
requirements.txt
report.pdf
```

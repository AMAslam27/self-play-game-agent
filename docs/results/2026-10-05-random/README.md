# Original training evaluation

This snapshot records the original 20,000-episode random-opponent training run,
`20261005T185241722454Z_a1f0bcd3`, with training seed 42 and base evaluation
seed 12345. The final greedy evaluation used 200 games per opponent and seat:

| Evaluation opponent | Agent seat | Games | Wins | Draws | Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | X | 200 | 198 | 2 | 0 |
| Random | O | 200 | 161 | 26 | 13 |
| Minimax | X | 200 | 0 | 200 | 0 |
| Minimax | O | 200 | 0 | 200 | 0 |

See the [resolved configuration](config.yaml) and
[evaluation history](evaluation.csv). The root README contains the newer
offline assessment of the same saved checkpoint across three evaluation seeds.

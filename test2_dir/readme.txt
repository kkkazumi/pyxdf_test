# graph drawing
## separate graph
`python3 test2.py`

select dir name

You can change the array of trigger just to make it easy to check.

## no-separate but with lines
`python3 test_line.py`

use the program to draw a graph with the line of triggers.

## Timeseries of HP values
`python3 HP_timeseries.py`
you will get a pic `000_HP_timeseries.*`
select `*.png` or `*.eps`

## statistics
to get statistic values of HP,
`python3 HP_statistics.py`
you will get `000_y_valuesHM.csv` and `000_y_valuesRB.csv`
HM is a data under the Human hint condition
RB is a data under the Robot hint condition

These two csv files are needed to generate pic when you execute `draw_HP_pic.py`



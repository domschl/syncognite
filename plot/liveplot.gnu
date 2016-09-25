set autoscale y
set autoscale y2
set ytics nomirror
set y2tics
set tics out
set xlabel "Epoch"
set ylabel "Loss"
set y2label "Validation accuracy"
set key left top
plot "log.txt" using 1:2 with points axes x1y1 title 'Loss', \
     "log.txt" using 1:3 with lines axes x1y1 title 'Mean Loss', \
     "log.txt" using 1:4 with lines axes x1y1 title 'Mean2 Loss', \
     "log.txt" using 1:5 with points axes x1y2 title 'Validation accuracy', \
     "log.txt" using 1:6 with lines axes x1y2 title 'Mean Validation accuracy'
pause 1
reread

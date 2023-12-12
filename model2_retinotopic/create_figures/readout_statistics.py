import numpy as np
import pandas as pd
import pingouin as pg

acc = pd.read_csv('results/intermediate/readout.csv')

print('Welch ANOVA')
aov_welch = pg.welch_anova(data=acc, dv='Test acc.', between='Population')#, detailed=False)
print(aov_welch)
print('\n')
      
print('Games-Howell post-hoc')
pwgh = pg.pairwise_gameshowell(data=acc, dv='Test acc.', between='Population').round(8)
print(pwgh)

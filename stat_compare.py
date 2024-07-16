# Imports
import pandas as pd
import matplotlib.pyplot as plt

#################################################################################################################

# Defining file paths
hh_path = 'ddaye-isp-datasets//hof_hitters.xlsx'
nhh_path = 'ddaye-isp-datasets/non-hof_hitters.xlsx'
hp_path = 'ddaye-isp-datasets/hof_pitchers.xlsx'
nhp_path = 'ddaye-isp-datasets/non-hof_pitchers.xlsx'

# Reading files into dataframes
hof_hitters = pd.read_excel(hh_path)
non_hof_hitters = pd.read_excel(nhh_path)
hof_pitchers = pd.read_excel(hp_path)
non_hof_pitchers = pd.read_excel(nhp_path)

#################################################################################################################

# Hitting data
hof_hitters_stats = hof_hitters[['Player','From','To','G','PA','H','1B','2B','3B','HR','RBI','BA','OBP','SLG','OPS','OPS+','TB']]
non_hof_hitters_stats = non_hof_hitters[['Player','From','To','G','PA','H','1B','2B','3B','HR','RBI','BA','OBP','SLG','OPS','OPS+','TB']]

# Pitching data
hof_pitchers_stats = hof_pitchers[['Player','From','To','G','ERA','IP','ERA+','FIP','WHIP','H9','BB9','SO9','SO/BB']]
non_hof_pitchers_stats = non_hof_pitchers[['Player','From','To','G','ERA','IP','ERA+','FIP','WHIP','H9','BB9','SO9','SO/BB']]

# # Setting criteria for to rule out pitchers who were generally not good at hitting
# # NOTE: The dataset for Non-HOF hitters all have minimum of GP <= 162.
p_hof_hitters = hof_hitters_stats[hof_hitters_stats['BA'] >= 0.240]
p_non_hof_hitters = non_hof_hitters_stats[non_hof_hitters_stats['G'] >= 50] # Redundant, just to create new dataframe for consistancy in naming scheme.

# Setting criteria for pitchers only (32+ G)
# NOTE: The dataset for Non-HOF Pitchers all have minimum of GP <= 32.
# NOTE: HOF Pitchers are filtered by 32+ G to remove position players who have pitched a few innings over their careers but are not pitchers.
p_hof_pitchers = hof_pitchers_stats[hof_pitchers_stats['G'] >= 32]
p_non_hof_pitchers = non_hof_pitchers_stats[non_hof_pitchers_stats['G'] >= 32] # Redundant, just to create new dataframe for consistancy in naming scheme.

#################################################################################################################

# Examining certain hitter stats
p_hof_hitters_ba = p_hof_hitters[['BA']]
p_non_hof_hitters_ba = p_non_hof_hitters[['BA']]

p_hof_hitters_obp = p_hof_hitters[['OBP']]
p_non_hof_hitters_obp = p_non_hof_hitters[['OBP']]

p_hof_hitters_slg = p_hof_hitters[['SLG']]
p_non_hof_hitters_slg = p_non_hof_hitters[['SLG']]

p_hof_hitters_ops = p_hof_hitters[['OPS']]
p_non_hof_hitters_ops = p_non_hof_hitters[['OPS']]

p_hof_hitters_opsplus = p_hof_hitters[['OPS+']]
p_non_hof_hitters_opsplus = p_non_hof_hitters[['OPS+']]

#################################################################################################################

# Examining certain pitcher stats
p_hof_pitchers_eraplus = p_hof_pitchers[['ERA+']]
p_non_hof_pitchers_eraplus = p_non_hof_pitchers[['ERA+']]

p_hof_pitchers_fip = p_hof_pitchers[['FIP']]
p_non_hof_pitchers_fip = p_non_hof_pitchers[['FIP']]

p_hof_pitchers_whip = p_hof_pitchers[['WHIP']]
p_non_hof_pitchers_whip = p_non_hof_pitchers[['WHIP']]

p_hof_pitchers_sobb = p_hof_pitchers[['SO/BB']]
p_non_hof_pitchers_sobb = p_non_hof_pitchers[['SO/BB']]

#################################################################################################################

# Plotting Hitters
fig, axs = plt.subplots(2,3, figsize=(15,8))

# BA for both overlayed
axs[0, 0].hist(p_hof_hitters_ba, bins=10, color='red', density=True, label='HoF')
axs[0, 0].hist(p_non_hof_hitters_ba, bins=50, color='teal', alpha=0.7, density=True, label='Non-HoF')
axs[0, 0].set_xlabel('Batting Average')
axs[0, 0].set_ylabel('Density of Players from Group')
axs[0, 0].set_title('Batting Average for HOF and Non-HOF Hitters (Higher is better)')
axs[0, 0].set_xlim(0,0.5)
axs[0, 0].legend()

# OBP for both overlayed
axs[0, 1].hist(p_hof_hitters_obp, bins=15, color='red', density=True, label='HoF')
axs[0, 1].hist(p_non_hof_hitters_obp, bins=30, color='teal', alpha=0.7, density=True, label='Non-HoF')
axs[0, 1].set_xlabel('On-base Percentage')
axs[0, 1].set_ylabel('Density of Players from Group')
axs[0, 1].set_title('OBP for HOF and Non-HOF Hitters (Higher is better)')
axs[0, 1].set_xlim(0,0.6)
axs[0, 1].legend()

# SLG for both overlayed
axs[0, 2].hist(p_hof_hitters_slg, bins=15, color='red', density=True, label='HoF')
axs[0, 2].hist(p_non_hof_hitters_slg, bins=50, color='teal', alpha=0.7, density=True, label='Non-HoF')
axs[0, 2].set_xlabel('Slugging Percentage')
axs[0, 2].set_ylabel('Density of Players from Group')
axs[0, 2].set_title('SLG for HOF and Non-HOF Hitters (Higher is better)')
axs[0, 2].set_xlim(0,0.75)
axs[0, 2].legend()

# OPS for both overlayed
axs[1, 0].hist(p_hof_hitters_ops, bins=15, color='red', density=True, label='HoF')
axs[1, 0].hist(p_non_hof_hitters_ops, bins=50, color='teal', alpha=0.7, density=True, label='Non-HoF')
axs[1, 0].set_xlabel('On-base plus Slugging')
axs[1, 0].set_ylabel('Density of Players from Group')
axs[1, 0].set_title('OPS for HOF and Non-HOF Hitters (Higher is better)')
axs[1, 0].set_xlim(0,1.2)
axs[1, 0].legend()

# OPS+ for both overlayed
axs[1, 1].hist(p_hof_hitters_opsplus, bins=15, color='red', density=True, label='HoF')
axs[1, 1].hist(p_non_hof_hitters_opsplus, bins=50, color='teal', alpha=0.7, density=True, label='Non-HoF')
axs[1, 1].set_xlabel('On-base plus Slugging Plus')
axs[1, 1].set_ylabel('Density of Players from Group')
axs[1, 1].set_title('OPS+ for HOF and Non-HOF Hitters (Higher is better)')
axs[1, 1].set_xlim(0,230)
axs[1, 1].legend()

# Removing empty axis
axs[1, 2].axis('off')

# Displaying the data
plt.tight_layout()
plt.savefig('ddaye-isp-files/hitters_compared.png')

#################################################################################################################

# Plotting Pitchers
fig, axs = plt.subplots(2,2, figsize=(10,8))

# ERA+ for both overlayed
axs[0, 0].hist(p_hof_pitchers_eraplus, bins=20, color='red', density=True, label='HoF')
axs[0, 0].hist(p_non_hof_pitchers_eraplus, bins=30, color='teal', alpha=0.6, density=True, label='Non-HoF')
axs[0, 0].set_xlabel('Adjusted Earned Run Average Plus')
axs[0, 0].set_ylabel('Density of Players from Group')
axs[0, 0].set_title('ERA+ for HOF and Non-HOF Pitchers (Higher is better)')
axs[0, 0].set_xlim(70,210)
axs[0, 0].legend()


# FIP for both overlayed
axs[0, 1].hist(p_hof_pitchers_fip, bins=12, color='red', density=True, label='HoF')
axs[0, 1].hist(p_non_hof_pitchers_fip, bins=30, color='teal', alpha=0.6, density=True, label='Non-HoF')
axs[0, 1].set_xlabel('Fielding Independent Pitching')
axs[0, 1].set_ylabel('Density of Players from Group')
axs[0, 1].set_title('FIP for HOF and Non-HOF Pitchers (Lower is better)')
axs[0, 1].set_xlim(1,6)
axs[0, 1].legend()

# WHIP for both overlayed
axs[1, 0].hist(p_hof_pitchers_whip, bins=12, color='red', density=True, label='HoF')
axs[1, 0].hist(p_non_hof_pitchers_whip, bins=30, color='teal', alpha=0.6, density=True, label='Non-HoF')
axs[1, 0].set_xlabel('Walks plus Hits per Inning Pitched')
axs[1, 0].set_ylabel('Density of Players from Group')
axs[1, 0].set_title('WHIP for HOF and Non-HOF Pitchers (Lower is better)')
axs[1, 0].set_xlim(0.8,2.0)
axs[1, 0].legend()

# SO/BB for both overlayed
axs[1, 1].hist(p_hof_pitchers_sobb, bins=12, color='red', density=True, label='HoF')
axs[1, 1].hist(p_non_hof_pitchers_sobb, bins=30, color='teal', alpha=0.6, density=True, label='Non-HoF')
axs[1, 1].set_xlabel('Strikeouts per Bases on Balls')
axs[1, 1].set_ylabel('Density of Players from Group')
axs[1, 1].set_title('SO/BB for HOF and Non-HOF Pitchers (Higher is better)')
axs[1, 1].set_xlim(0.4,7.6)
axs[1, 1].legend()

# Displaying the data
plt.tight_layout()
plt.savefig('ddaye-isp-files/pitchers_compared.png')
plt.show
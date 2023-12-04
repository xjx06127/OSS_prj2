import pandas as pd

data = pd.read_csv("2019_kbo_for_kaggle_v2.csv")

# Print the top 10 players in hits (안타, H), batting average (타율, avg), homerun (홈런, HR)
# and onbase percentage (출루율, OBP) for each year from 2015 to 2018.
print("--------------problem 1--------------")
for year in range(2015, 2019):
    year_data = data[data['year'] == year] # year에 따른 데이터 먼저 추출
    sorted_data_H = year_data.sort_values(by='H', ascending=False)
    print(sorted_data_H[['batter_name', 'year', 'H']].head(10))

    sorted_data_avg = year_data.sort_values(by='avg', ascending=False)
    print(sorted_data_avg[['batter_name', 'year', 'avg']].head(10))

    sorted_data_HR = year_data.sort_values(by='HR', ascending=False)
    print(sorted_data_HR[['batter_name', 'year', 'HR']].head(10))

    sorted_data_OBP = year_data.sort_values(by='OBP', ascending=False)
    print(sorted_data_OBP[['batter_name', 'year', 'OBP']].head(10))

# Print the player with the highest war (승리 기여도) by position (cp) in 2018.
print("--------------problem 2--------------")
year_data_2018 = data[data['year'] == 2018] # 2018년 데이터 추출
position = pd.Series(['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수'])
for pos in position:
    pos_data = year_data_2018[year_data_2018['cp'] == pos] # 2018년, cp에 따른 데이터 추출
    sorted_data_war = pos_data.sort_values(by="war", ascending=False)
    print(sorted_data_war[['batter_name', 'year', 'war', 'cp']].head(1))

# Among R (득점), H (안타), HR (홈런), RBI (타점), SB (도루), war (승리 기여도), avg (타율), OBP
# (출루율), and SLG (장타율), which has the highest correlation with salary (연봉)?
print("--------------problem 3--------------")
data_others = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']]
data_salary = data['salary']
correlation = data_others.corrwith(data_salary)

print(correlation)
print(correlation.idxmax() + " has the highest correlation with salary.")


import os
import io
import csv
import time
import lxml
import streamlit as st

from datetime import datetime 
import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from sklearn.preprocessing import MinMaxScaler
import re 

from streamlit_gsheets import GSheetsConnection
import google.generativeai as genai
# import pandasql as psql

warnings.filterwarnings('ignore')
st.set_page_config(page_title="호시탐탐 기록실")
#st.title('성남리그(토요) 기록실') ### title

## 성남리그 팀 딕셔너리 및 영문 그래프용 딕셔너리 & 리스트
team_id_dict_2025rkC = {
    '코메츠 호시탐탐': 7984,   '보성야구단': 15977,     '데빌베어스(Devil Bears)': 19135,    'FA Members': 13621,
    'Team 야놀자': 39918,    '슈퍼스타즈': 23785,    'MANNA ECCLESIA': 43133,    '성남야구선수촌': 7072,    
    '라이노즈': 41326,    '에자이갑스': 23042,    '실버서울 야구단': 15753,    '야호 이겨스': 42160, '마자야지': 19163, '다이아몬스터': 39783,    'HEAT': 18414
}

team_name_dict_2025rkC = {
    '코메츠 호시탐탐': 'HOSHI',   '보성야구단': 'Bosung', '데빌베어스(Devil Bears)': 'DevilBears', 'FA Members': 'FAMembers',
    'Team 야놀자': 'TeamYnj', '슈퍼스타즈': 'Superstars','MANNA ECCLESIA': 'MANNAECCLESIA', '성남야구선수촌': 'SeongnamYgssc',    
    '라이노즈': 'Rhinos', '에자이갑스': 'EisaiGabs', '실버서울 야구단': 'SilverSeoul', '야호 이겨스': 'Yaho', '마자야지': 'MajaYaji', '다이아몬스터': 'Diamonster', 'HEAT': 'HEAT'
}

team_id_dict = team_id_dict_2025rkC #| team_id_dict_2025miB
team_name_dict = team_name_dict_2025rkC #| team_name_dict_2025miB

# 타자 데이터프레임 df에 적용할 자료형 / 컬럼명 딕셔너리 정의
hitter_data_types = {
    '성명': 'str', '배번': 'str', '타율': 'float', '경기': 'int', '타석': 'int', '타수': 'int',
    '득점': 'int', '총안타': 'int', '1루타': 'int', '2루타': 'int', '3루타': 'int', '홈런': 'int',
    '루타': 'int', '타점': 'int', '도루': 'int', '도실(도루자)': 'int', '희타': 'int', '희비': 'int',
    '볼넷': 'int', '고의4구': 'int', '사구': 'int', '삼진': 'int', '병살': 'int', '장타율': 'float',
    '출루율': 'float', '도루성공률': 'float', '멀티히트': 'int', 'OPS': 'float', 'BB/K': 'float',
    '장타/안타': 'float', '팀': 'str'
}
hitter_data_KrEn = {
    '성명': 'Name', '배번': 'No', '타율': 'AVG', '경기': 'G', '타석': 'PA', '타수': 'AB',
    '득점': 'R', '총안타': 'H', '1루타': '1B', '2루타': '2B', '3루타': '3B', '홈런': 'HR',
    '루타': 'TB', '타점': 'RBI', '도루': 'SB', '도실(도루자)': 'CS', '희타': 'SH', '희비': 'SF',
    '볼넷': 'BB', '고의4구': 'IBB', '사구': 'HBP', '삼진': 'SO', '병살': 'DP', '장타율': 'SLG', '출루율': 'OBP', '도루성공률': 'SB%', '멀티히트': 'MHit', 'OPS': 'OPS', 'BB/K': 'BB/K',
    '장타/안타': 'XBH/H', '팀': 'Team'
}
hitter_data_EnKr = {'Name': '성명', 'No': '배번', 'AVG': '타율', 'G': '경기', 'PA': '타석', 'AB': '타수', 'R': '득점', 
                    'H': '총안타', '1B': '1루타', '2B': '2루타', '3B': '3루타', 'HR': '홈런', 'TB': '루타', 'RBI': '타점', 
                    'SB': '도루', 'CS': '도실', 'SH': '희타', 'SF': '희비', 'BB': '볼넷', 'IBB': '고의4구', 'HBP': '사구', 'SO': '삼진', 'DP': '병살', 'SLG': '장타율', 'OBP': '출루율', 'SB%': '도루성공률', 'MHit': '멀티히트', 'OPS': 'OPS', 'BB/K': 'BB/K', 'XBH/H': '장타/안타', 'Team': '팀'}
# 투수 데이터프레임 df_pitcher에 적용할 자료형 / 컬럼명 딕셔너리 정의
pitcher_data_types = {
    '성명': 'str', '배번': 'str', '방어율': 'float', '경기수': 'int', '승': 'int', '패': 'int', '세': 'int',
    '홀드': 'int', '승률': 'float', '타자': 'int', '타수': 'int', '투구수': 'int', '이닝': 'float',
    '피안타': 'int', '피홈런': 'int', '희타': 'int', '희비': 'int', '볼넷': 'int', '고의4구': 'int',
    '사구': 'int', '탈삼진': 'int', '폭투': 'int', '보크': 'int', '실점': 'int', '자책점': 'int',
    'WHIP': 'float', '피안타율': 'float', '탈삼진율': 'float', '팀': 'str'
}
pitcher_data_KrEn = {
    '성명': 'Name', '배번': 'No', '방어율': 'ERA', '경기수': 'G', '승': 'W', '패': 'L', '세': 'SV',
    '홀드': 'HLD', '승률': 'WPCT', '타자': 'BF', '타수': 'AB', '투구수': 'P', '이닝': 'IP',
    '피안타': 'HA', '피홈런': 'HR', '희타': 'SH', '희비': 'SF', '볼넷': 'BB', '고의4구': 'IBB',
    '사구': 'HBP', '탈삼진': 'SO', '폭투': 'WP', '보크': 'BK', '실점': 'R', '자책점': 'ER',
    'WHIP': 'WHIP', '피안타율': 'BAA', '피장타율': 'SLG', '피출루율': 'OBP', '피OPS' : 'OPS', '탈삼진율': 'K9', '팀': 'Team'
}
pitcher_data_EnKr = {'Name': '성명', 'No': '배번', 'ERA': '방어율', 'G': '경기수', 'W': '승', 'L': '패', 'SV': '세', 'HLD': '홀드', 'WPCT': '승률', 
                     'BF': '타자', 'AB': '타수', 'P': '투구수', 'IP': '이닝', 'HA': '피안타', 'HR': '피홈런', 'SH': '희타', 'SF': '희비', 'BB': '볼넷', 'IBB': '고의4구', 'HBP': '사구', 
                     'SO': '탈삼진', 'WP': '폭투', 'BK': '보크', 'R': '실점', 'ER': '자책점', 'WHIP': 'WHIP', 'BAA': '피안타율', 'SLG':'피장타율', 'OBP':'피출루율', 'OPS' : '피OPS', 
                     'K9': '탈삼진율', 'Team': '팀'}

################################################################
## User def functions
################################################################
def create_heatmap(data, cmap, input_figsize=(10, 7)):
    fig, ax = plt.subplots(figsize=input_figsize)
    sns.heatmap(
        data, annot=True, fmt=".0f", cmap=cmap,
        annot_kws={'color': 'black'},
        yticklabels=data.index, cbar=False,
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel('')
    fig.tight_layout()
    return fig  # ✅ Figure 객체 반환

# 경기 결과에 따라 각 tr에 style 적용
def color_row_by_result(row_html: str) -> str:
    if "<td>승</td>" in row_html:
        return row_html.replace("<tr>", '<tr style="background-color: #d4f7d4;">')  # 연초록
    elif "<td>콜드승</td>" in row_html:
        return row_html.replace("<tr>", '<tr style="background-color: #d4f7d4;">')  # 연초록        
    elif "<td>패</td>" in row_html:
        return row_html.replace("<tr>", '<tr style="background-color: #fce2e2;">')  # 연분홍
    elif "<td>콜드패</td>" in row_html:
        return row_html.replace("<tr>", '<tr style="background-color: #fce2e2;">')  # 연분홍
    elif "<td>경기전</td>" in row_html:
        return row_html.replace("<tr>", '<tr style="background-color: #f0f0f0;">')  # 연회색
    return row_html  # 변화 없음

# tbody 내부만 찾아서 각 tr 가공
def apply_row_styling(html: str) -> str:
    tbody_content = re.search(r"<tbody>(.*?)</tbody>", html, re.DOTALL)
    if not tbody_content:
        return html

    tbody_html = tbody_content.group(1)
    styled_rows = []
    for row_html in re.findall(r"<tr>.*?</tr>", tbody_html, re.DOTALL):
        styled_row = color_row_by_result(row_html)
        styled_rows.append(styled_row)

    styled_tbody = "<tbody>\n" + "\n".join(styled_rows) + "\n</tbody>"
    return re.sub(r"<tbody>.*?</tbody>", styled_tbody, html, flags=re.DOTALL)

def format_cell(x):
    # 정수는 그대로, float는 소수 4자리까지
    if isinstance(x, int):
        return f"{x}"
    elif isinstance(x, float) and x.is_integer():
        return f"{int(x)}"
    elif isinstance(x, float):
        return f"{x:.3f}"
    else:
        return x


# 테이블 CSS
table_style = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 10px;
            background-color: white; /* 다크모드에서도 흰 배경 */
            color: black; /* 글자 검정색 */
        }
        th, td {
            border: 1px solid #999;
            padding: 4px 6px;
            text-align: center;
        }
        th {
            background-color: #e6e6e6;  /* 약간 어두운 회색 */
            font-weight: bold;
        }
    </style>
"""

table_style_12px = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;  /* 열 너비를 동일하게 고정 */
            font-size: 12px;
            background-color: white; /* 다크모드에서도 흰 배경 */
            color: black; /* 글자 검정색 */
        }
        th, td {
            border: 1px solid #999;
            padding: 4px 6px;
            text-align: center;
            word-wrap: break-word;  /* 내용이 길면 줄바꿈 */
        }
        th {
            background-color: #e6e6e6;  /* 약간 어두운 회색 */
            font-weight: bold;
        }
    </style>
"""

def data_to_text(data, max_rows: int = 30) -> str:
    # 딕셔너리인 경우 처리
    if isinstance(data, dict):
        # 딕셔너리를 DataFrame으로 변환
        # 딕셔너리 구조에 따라 다르게 처리
        if all(isinstance(v, (list, tuple)) for v in data.values()):
            # 키가 열 이름이고 값이 리스트인 경우 (일반적인 형태)
            df = pd.DataFrame(data)
        else:
            # 중첩된 딕셔너리나 다른 형태의 딕셔너리
            df = pd.DataFrame([data])
        
        return data_to_text(df, max_rows)
    
    # DataFrame인 경우 처리
    elif isinstance(data, pd.DataFrame):
        if len(data) > max_rows:
            data = data.head(max_rows)
        return data.to_csv(index=False)
    
    # 리스트인 경우 처리 (추가 기능)
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            # 딕셔너리 리스트인 경우
            df = pd.DataFrame(data)
            return data_to_text(df, max_rows)
        else:
            # 일반 리스트인 경우
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 행 제한 적용
            if len(data) > max_rows:
                data = data[:max_rows]
                
            for item in data:
                writer.writerow([item])
            
            return output.getvalue()
    
    # 그 외 타입인 경우
    else:
        return str(data)

@st.cache_data
def load_data(team_name, team_id, default_year):
    urls = {
        'hitter': f"http://www.gameone.kr/club/info/ranking/hitter?club_idx={team_id}&kind=&season={default_year}",
        'pitcher': f"http://www.gameone.kr/club/info/ranking/pitcher?club_idx={team_id}&kind=&season={default_year}"
    }
    results = {'hitter': [], 'pitcher': []}
    for key, url in urls.items():
        response = requests.get(url)
        tables = pd.read_html(response.text)
        for table in tables:
            extracted_df = table['이름'].str.extract(r"(\w+)\((\d+)\)")
            extracted_df.columns = ['성명', '배번']
            extracted_df['배번'] = extracted_df['배번'].astype(int)
            table = pd.concat([extracted_df, table.drop(['이름'], axis=1)], axis=1)
            # 컬럼명 변경
            if '게임수' in table.columns:
                if key == 'hitter':
                    table.rename(columns={'게임수': '경기'}, inplace=True)
                else:
                    table.rename(columns={'게임수': '경기수'}, inplace=True)

            table['팀'] = team_name  # 팀 이름 컬럼 추가
            table = table.drop('순위', axis = 1)
            table.columns = [col.replace(" ", "") for col in table.columns]
            results[key].append(table)
    return {'hitter': pd.concat(results['hitter'], ignore_index=True), 
            'pitcher': pd.concat(results['pitcher'], ignore_index=True)}

# 화면 최상단을 3개의 컬럼으로 나누기
top_col1, top_col2, top_col3 = st.columns(3)

####################################
#### 일정표 준비
####################################
# 강조할 팀명
highlight_team = '코메츠 호시탐탐'

# 스타일 적용 함수
def highlight_team_name(team_name, highlight_target):
    if team_name == highlight_target:
        return f"<span style='color: skyblue; font-weight: bold;'>{team_name}</span>"
    return team_name

# 일정표 크롤링 & 다음경기 출력
with top_col1:
    this_year = 2025
    # 일정표 URL 설정
    schd_url = f"http://www.gameone.kr/club/info/schedule/table?club_idx=7984&kind=&season={this_year}"
    # HTTP GET 요청
    response = requests.get(schd_url)
    response.raise_for_status()  # 요청이 성공했는지 확인

    # BeautifulSoup을 이용하여 HTML 파싱
    soup = BeautifulSoup(response.content, 'html.parser')

    # 테이블 찾기
    table = soup.find('table', {'class': 'game_table'})  # 테이블의 클래스를 확인하고 지정하세요

    # 테이블 헤더 추출
    headers = [header.text.strip() for header in table.find_all('th')]

    # 테이블 데이터 추출
    rows = []
    for row in table.find_all('tr')[1:]:  # 첫 번째 행은 헤더이므로 제외
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)

    # pandas DataFrame 생성
    df_schd = pd.DataFrame(rows, columns=headers)
    df_schd = df_schd.sort_values('일시').reset_index(drop=True)
    data = df_schd.게임.str.split('\n').tolist()
    # 최대 열 개수 확인
    max_columns = max(len(row) for row in data)
    # 열 이름 설정
    column_names = [f"col{i+1}" for i in range(max_columns)]
    # DataFrame 생성
    df_team = pd.DataFrame(data, columns=column_names).drop(['col3', 'col4', 'col5'], axis =1)
    # DataFrame 출력
    df_schd2 = pd.concat([df_schd.drop(['게임', '분류'], axis =1), df_team], axis = 1)
    # 열 갯수가 6개일 경우, '6' 컬럼을 추가
    if df_schd2.shape[1] == 6:
        df_schd2['6'] = ''  # '' 값을 가진 빈 컬럼을 추가    
    df_schd2.columns = ['일시', '구장', '결과', '선공', '선공점수', '후공', '후공점수']
    df_schd2.구장 = df_schd2.구장.str.replace('야구장', '')
    first_called = df_schd2.선공점수.str.contains('콜드승')
    second_called = df_schd2.후공점수.str.contains('콜드승')
    df_schd2.선공점수 = df_schd2.선공점수.str.replace('콜드승 ', '').str.replace('기권승 ', '').str.replace('몰수승 ', '').replace(r'^\s*$', pd.NA, regex=True).fillna(0).astype('int')  #.replace('', 0).astype('int')
    df_schd2.후공점수 = df_schd2.후공점수.str.replace('콜드승 ', '').str.replace('기권승 ', '').str.replace('몰수승 ', '').replace(r'^\s*$', pd.NA, regex=True).fillna(0).astype('int')  #.replace('', 0).astype('int')
    df_schd2['Result'] = ''
    tmp_result = list()
    for i in range(df_schd2.shape[0]):
        # print(i, first_called[i], second_called[i])
        if df_schd2.iloc[i]['선공점수'] > df_schd2.iloc[i]['후공점수']:
            if first_called[i]:
                result = df_schd2.iloc[i]['선공'] + '_콜드승'    
            else :
                result = df_schd2.iloc[i]['선공'] + '_승'
        elif df_schd2.iloc[i]['선공점수'] < df_schd2.iloc[i]['후공점수']:
            if second_called[i]:
                result = df_schd2.iloc[i]['후공'] + '_콜드승'
            else:
                result = df_schd2.iloc[i]['후공'] + '_승'
            # print(i, result)
        elif (df_schd2.iloc[i]['결과'] != '게임대기') & (df_schd2.iloc[i]['선공점수'] == df_schd2.iloc[i]['후공점수']):
            result = '무'
            # print(i, result)
        else:
            result = '경기전'
            # print(i, result)
        tmp_result.append(result)

    df_schd2['Result'] = tmp_result
    df_schd2.loc[df_schd2['Result'].str.contains('호시탐탐_콜드승'), 'Result'] = '콜드승'
    df_schd2.loc[df_schd2['Result'].str.contains('호시탐탐_승'), 'Result'] = '승'
    df_schd2.loc[df_schd2['Result'].str.contains('_승'), 'Result'] = '패'
    df_schd2.loc[df_schd2['Result'].str.contains('_콜드승'), 'Result'] = '콜드패'

    df_schd2 = df_schd2.drop('결과', axis = 1)
    df_schd2.columns = ['일시', '구장', '선공', '선', '후공', '후', '결과']

    next_game = df_schd2.loc[df_schd2['결과'] == '경기전', ['일시', '구장', '선공', '후공']].head(1).reset_index(drop=True)
    next_game_teamname = ((next_game['선공'] + next_game['후공']).str.replace('코메츠 호시탐탐', ''))[0]
    # 선공/후공 팀명에 스타일 적용
    away_team = highlight_team_name(next_game['선공'][0], highlight_team)
    home_team = highlight_team_name(next_game['후공'][0], highlight_team)
    # 전체 문장 구성
    markdown_text = f"""
        [NEXT] {next_game['일시'][0]} [{next_game['구장'][0]}]  
        {away_team} vs {home_team}
    """
    # 출력
    st.markdown(markdown_text, unsafe_allow_html=True)
with top_col2:
    ## 년도 설정
    year_list = [2025, 2024, 2023, 2022, 2021]
    default_year = st.selectbox('년도 선택', year_list, index = 0, key = 'year_selectbox')
with top_col3:
    # 전체 팀 목록
    team_list_all = list(team_id_dict.keys())

    # highlight_team & next_game_teamname 제외한 나머지 팀들 정렬
    next_game = df_schd2.loc[df_schd2['결과'] == '경기전', ['일시', '구장', '선공', '후공']].head(1).reset_index(drop=True)
    next_game_teamname = ((next_game['선공'] + next_game['후공']).str.replace(highlight_team, ''))[0] if not next_game.empty else None

    # 나머지 팀들 정렬 (highlight_team과 next_game_teamname 제외)
    other_teams = sorted(
        team for team in team_list_all
        if team != highlight_team and team != next_game_teamname
    )

    # 최종 정렬된 팀 리스트 (우리팀 -> 다음 경기 상대팀 -> 나머지 팀들)
    team_list = [highlight_team]
    if next_game_teamname:
        team_list.append(next_game_teamname)
    team_list.extend(other_teams)

    team_name = st.selectbox('팀 선택', team_list, key = 'selbox_team_entire')
    team_id = team_id_dict[team_name]
    rank_calc_include_teams = list(team_id_dict.keys())
    team_groupname = "토요 마이너B"       


################################################################
## Data Loading
################################################################
sn_standings_url = 'http://www.gameone.kr/league/record/rank?lig_idx=10373'

try:        # Create GSheets connection AND Load Data from google sheets 
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Read Google WorkSheet as DataFrame
    df_hitter = conn.read(worksheet="df_hitter_{}".format(default_year))
    df_pitcher = conn.read(worksheet="df_pitcher_{}".format(default_year))
    st.write()    
    time.sleep(1.5)   
    st.toast(f'Loaded Data from Cloud!', icon='✅')
except Exception as e: ## 만약 csv 파일 로드에 실패하거나 에러가 발생하면 병렬로 데이터 로딩
    st.error(f"Failed to read data from drive: {e}", icon="🚨") 
    hitters = []
    pitchers = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_data, team_name, team_id, default_year): team_name for team_name, team_id in team_id_dict.items()}
        for future in as_completed(futures):
            try:
                result = future.result()
                hitters.append(result['hitter'])
                pitchers.append(result['pitcher'])
            except Exception as exc:
                print(f'Team {futures[future]} generated an exception: {exc}')
    # 모든 데이터를 각각의 데이터프레임으로 합침
    final_hitters_data = pd.concat(hitters, ignore_index=True)
    final_pitchers_data = pd.concat(pitchers, ignore_index=True)

    # 데이터프레임 df의 컬럼 자료형 설정
    df_hitter = final_hitters_data.astype(hitter_data_types)
    # 타자 데이터프레임 컬럼명 영어로
    df_hitter.columns = ['Name', 'No', 'AVG', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 
                         'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'SLG', 'OBP', 'SB%', 'MHit', 
                         'OPS', 'BB/K', 'XBH/H', 'Team']

    final_pitchers_data.loc[final_pitchers_data.방어율 == '-', '방어율'] = np.nan

    # 투수 데이터프레임 df_pitcher의 컬럼 자료형 설정
    df_pitcher = final_pitchers_data.astype(pitcher_data_types)
    # 투수 데이터프레임 컬럼명 영어로
    df_pitcher.columns = ['Name', 'No', 'ERA', 'G', 'W', 'L', 'SV', 'HLD', 'WPCT', 
                          'BF', 'AB', 'P', 'IP', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'WP', 'BK', 
                        'R', 'ER', 'WHIP', 'BAA', 'K9', 'Team']
    # IP 컬럼을 올바른 소수 형태로 변환
    df_pitcher['IP'] = df_pitcher['IP'].apply(lambda x: int(x) + (x % 1) * 10 / 3).round(2)

    # Create GSheets connection
    conn = st.connection("gsheets", type=GSheetsConnection)

    # click button to update worksheet / This is behind a button to avoid exceeding Google API Quota
    if st.button("Loading Dataset"):
        try:
            df_hitter = conn.create(worksheet="df_hitter_{}".format(default_year), data=df_hitter)
        except Exception as e:
            st.error(f"Failed to save df_hitter: {e}", icon="🚨")        
            df_hitter = conn.update(worksheet="df_hitter_{}".format(default_year), data=df_hitter)
            st.toast('Updete Hitter Data from Web to Cloud!', icon='💾')
        
        try:
            df_pitcher = conn.create(worksheet="df_pitcher_{}".format(default_year), data=df_pitcher)
        except Exception as e:
            st.error(f"Failed to save df_pitcher: {e}", icon="🚨")        
            df_pitcher = conn.update(worksheet="df_pitcher_{}".format(default_year), data=df_pitcher)               
            st.toast('Updete Pitcher Data from Web to Cloud!', icon='💾')
        time.sleep(2)
        st.toast('Saved Data from Web to Cloud!', icon='💾')

################################################################
## DATASET PREPARE
################################################################
df_hitter = df_hitter.loc[df_hitter['Team'].isin(rank_calc_include_teams)].copy().reset_index(drop=True)
df_pitcher = df_pitcher.loc[df_pitcher['Team'].isin(rank_calc_include_teams)].copy().reset_index(drop=True)

# 팀별 데이터셋 그룹바이로 준비
## 1) 타자 데이터셋 / 출력시 열 순서 변경
rank_by_cols_h_sorted = ['Team', 'AVG', 'PA', 'AB', 'H', 'RBI', 'R', 'OBP', 'SLG', 'OPS', 'SO', 'BB', 
                         'SB', 'MHit', '1B', '2B', '3B', 'HR', 'TB', 'CS', 'SH', 'SF', 'IBB', 'HBP', 'DP']
hitter_sumcols = ['PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'MHit']
hitter_grpby = df_hitter.loc[df_hitter['Team'].isin(rank_calc_include_teams), hitter_sumcols + ['Team']].groupby('Team').sum().reset_index()

# 팀명을 기준으로 우리팀이 맨위에 오도록 설정
hitter_grpby = hitter_grpby.sort_values(by='Team')        # 1. Team 명 기준 오름차순 정렬
target = hitter_grpby[hitter_grpby['Team'].str.contains(highlight_team)]  # 2. 특정 문자열이 있는 행 필터링
others = hitter_grpby[~hitter_grpby['Team'].str.contains(highlight_team)]         # 3. 나머지 행 필터링
hitter_grpby = pd.concat([target, others], ignore_index=True)  # 4. 두 데이터프레임을 위에서 아래로 concat

# 타율(AVG), 출루율(OBP), 장타율(SLG), OPS 계산 & 반올림
hitter_grpby['AVG'] = (hitter_grpby['H'] / hitter_grpby['AB']).round(3)
hitter_grpby['OBP'] = ((hitter_grpby['H'] + hitter_grpby['BB'] + hitter_grpby['HBP']) / (hitter_grpby['AB'] + hitter_grpby['BB'] + hitter_grpby['HBP'] + hitter_grpby['SF'])).round(3)
hitter_grpby['SLG'] = (hitter_grpby['TB'] / hitter_grpby['AB']).round(3)
hitter_grpby['OPS'] = (hitter_grpby['OBP'] + hitter_grpby['SLG']).round(3)

# 'Team' 컬럼 바로 다음에 계산된 컬럼들 삽입
for col in ['OPS', 'SLG', 'OBP', 'AVG']:
    team_idx = hitter_grpby.columns.get_loc('Team') + 1
    hitter_grpby.insert(team_idx, col, hitter_grpby.pop(col))

# rank_by_ascending, rank_by_descending columns 
rank_by_ascending_cols_h = ['SO', 'DP', 'CS'] # 낮을수록 좋은 지표들
rank_by_descending_cols_h = ['AVG', 'OBP', 'SLG', 'OPS', 'PA', 'AB', 'R', 'H', 'MHit', 
            '1B', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'SH', 'SF', 'BB', 'IBB', 'HBP'] # 높을수록 좋은 지표들
# st.dataframe(hitter_grpby.loc[:, rank_by_cols_h_sorted].rename(columns = hitter_data_EnKr, inplace=False), use_container_width = True, hide_index = True)
hitter_grpby_rank = pd.concat([
                                hitter_grpby.Team, 
                                hitter_grpby[rank_by_descending_cols_h].rank(method = 'min', ascending=False),
                                hitter_grpby[rank_by_ascending_cols_h].rank(method = 'min', ascending=True)
                            ], axis = 1)
hitter_grpby_rank = hitter_grpby_rank.loc[:, rank_by_cols_h_sorted] 

## 2) 투수 데이터셋
rank_by_cols_p_sorted = ['Team', 'IP', 'ERA', 'WHIP', 'H/IP', 'BB/IP', 'SO/IP', 'BAA', 'OBP', 'G', 'W', 'L', 'SV', 'HLD', 
                            'SO', 'BF', 'AB', 'P', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'R', 'ER', 'K9']  
if df_pitcher.shape[0] > 0 : # pitcher data exists
    # 출력시 열 순서 변경
    # st.subheader('전체투수 [{}명]'.format(df_pitcher.shape[0]))
    pitcher_sumcols = df_pitcher.select_dtypes(include=['int64', 'float64']).columns.tolist() # + ['IP'] # Sum 컬럼 선택
    pitcher_sumcols = [col for col in pitcher_sumcols if col != 'No'] # No 열 제외하기

    # 이닝당 삼진/볼넷/피안타 계산 (예제로 삼진(K), 볼넷(BB), 피안타(HA) 컬럼 필요)
    if 'SO' in df_pitcher.columns and 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
        df_pitcher['SO/IP'] = (df_pitcher['SO'] / df_pitcher['IP']).round(2)
        df_pitcher['BB/IP'] = (df_pitcher['BB'] / df_pitcher['IP']).round(2)
        df_pitcher['H/IP'] = (df_pitcher['HA'] / df_pitcher['IP']).round(2)
    
    # WHIP 계산: (볼넷 + 피안타) / 이닝
    if 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
        df_pitcher['WHIP'] = ((df_pitcher['BB'] + df_pitcher['HA']) / df_pitcher['IP']).round(3)
        df_pitcher['OBP'] = (df_pitcher['HA'] + df_pitcher['BB'] + df_pitcher['HBP']) / (df_pitcher['AB'] + df_pitcher['BB'] + df_pitcher['HBP'] + df_pitcher['SF'])
        # df_pitcher['SLG'] = (df_pitcher['HA'] + df_pitcher['2B']*2 + df_pitcher['3B']*3 + df_pitcher['HR']*4) / df_pitcher['AB']
        # df_pitcher['OPS'] = df_pitcher['OBP'] + df_pitcher['SLG']

    # None, '', '-'를 NaN으로 변환
    df_pitcher = df_pitcher.replace({None: np.nan, '': np.nan, '-': np.nan}) #, inplace=True)
    # 필요한 컬럼을 정의
    p_required_columns = ['No', 'Name'] + rank_by_cols_p_sorted
    # 존재하는 컬럼만 선택
    p_existing_columns = [col for col in p_required_columns if col in df_pitcher.columns]
    team_p_existing_columns = [col for col in rank_by_cols_p_sorted if col in df_pitcher.columns]

    pitcher_grpby = df_pitcher.loc[df_pitcher['Team'].isin(rank_calc_include_teams), 
                                    ['Team']+pitcher_sumcols].groupby('Team')[pitcher_sumcols].sum().reset_index()  # 팀별 합계 (인덱스가 팀명)
    
    # 팀명을 기준으로 우리팀이 맨위에 오도록 설정
    pitcher_grpby = pitcher_grpby.sort_values(by='Team')        # 1. Team 명 기준 오름차순 정렬
    target = pitcher_grpby[pitcher_grpby['Team'].str.contains(highlight_team)]  # 2. 특정 문자열이 있는 행 필터링
    others = pitcher_grpby[~pitcher_grpby['Team'].str.contains(highlight_team)]         # 3. 나머지 행 필터링
    pitcher_grpby = pd.concat([target, others], ignore_index=True)  # 4. 두 데이터프레임을 위에서 아래로 concat
    
    # 파생 변수 추가
    # 방어율(ERA) 계산: (자책점 / 이닝) * 9 (예제로 자책점과 이닝 컬럼 필요)
    if 'ER' in df_pitcher.columns and 'IP' in df_pitcher.columns:
        pitcher_grpby['ERA'] = ((pitcher_grpby['ER'] / pitcher_grpby['IP']) * 9).round(2)

    # 이닝당 삼진/볼넷/피안타 계산 (예제로 삼진(K), 볼넷(BB), 피안타(HA) 컬럼 필요)
    if 'SO' in df_pitcher.columns and 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
        pitcher_grpby['SO/IP'] = (pitcher_grpby['SO'] / pitcher_grpby['IP']).round(2)
        pitcher_grpby['BB/IP'] = (pitcher_grpby['BB'] / pitcher_grpby['IP']).round(2)
        pitcher_grpby['H/IP'] = (pitcher_grpby['HA'] / pitcher_grpby['IP']).round(2)
        pitcher_grpby['K9'] = (pitcher_grpby['SO/IP'] * 9)

    # WHIP 계산: (볼넷 + 피안타) / 이닝
    if 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
        pitcher_grpby['WHIP'] = ((pitcher_grpby['BB'] + pitcher_grpby['HA']) / pitcher_grpby['IP']).round(3)
        pitcher_grpby['BAA'] = (pitcher_grpby['HA'] / pitcher_grpby['AB']).round(3)
        pitcher_grpby['OBP'] = (pitcher_grpby['HA'] + pitcher_grpby['BB'] + pitcher_grpby['HBP']) / (pitcher_grpby['AB'] + pitcher_grpby['BB'] + pitcher_grpby['HBP'] + pitcher_grpby['SF']).round(3)
        # pitcher_grpby['SLG'] = (pitcher_grpby['HA'] + pitcher_grpby['2B']*2 + pitcher_grpby['3B']*3 + pitcher_grpby['HR']*4) / pitcher_grpby['AB']
        # pitcher_grpby['OPS'] = pitcher_grpby['OBP'] + pitcher_grpby['SLG']

    # 'Team' 컬럼 바로 다음에 계산된 컬럼들 삽입
    new_cols = ['K/IP', 'BB/IP', 'H/IP', 'WHIP', 'ERA', 'BAA', 'OBP'] # , 'OPS', 'OBP', 'SLG']
    for col in new_cols:
        if col in pitcher_grpby.columns:
            team_idx = pitcher_grpby.columns.get_loc('Team') + 1
            pitcher_grpby.insert(team_idx, col, pitcher_grpby.pop(col))

    # 결과 확인
    # rank_by_ascending, rank_by_descending columns  
    rank_by_ascending_cols_p = ['ERA', 'WHIP', 'H/IP', 'BB/IP', 'BAA', 'OBP', 'BF', 'AB', 'P', 'HA', 'HR', 
                                'SH', 'SF', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'R', 'ER'] # 낮을수록 좋은 지표들
    rank_by_descending_cols_p = ['IP', 'G', 'W', 'L', 'SV', 'HLD', 'SO', 'SO/IP', 'K9'] # 높을수록 좋은 지표들

    pitcher_grpby_rank = pd.concat([
                                    pitcher_grpby.Team, 
                                    pitcher_grpby[rank_by_descending_cols_p].rank(method = 'min', ascending=False),
                                    pitcher_grpby[rank_by_ascending_cols_p].rank(method = 'min', ascending=True)
                                ], axis = 1)
    pitcher_grpby_rank = pitcher_grpby_rank.loc[:, team_p_existing_columns]

##################################
# 2022-2025 누적지표 데이터 계산
##################################
tot_df_hitter = pd.DataFrame()
tot_df_pitcher = pd.DataFrame()
for i in year_list:
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Read Google WorkSheet as DataFrame
    tmp_df_hitter = conn.read(worksheet="df_hitter_{}".format(i))
    tmp_df_hitter['Year'] = i
    tot_df_hitter = pd.concat([tot_df_hitter, tmp_df_hitter], axis = 0).reset_index(drop=True)
    
    tmp_df_pitcher = conn.read(worksheet="df_pitcher_{}".format(i))
    tmp_df_pitcher['Year'] = i
    tot_df_pitcher = pd.concat([tot_df_pitcher, tmp_df_pitcher], axis = 0).reset_index(drop=True)

# 타자 누적합 가능한 컬럼
sum_cols_hitter = [ "G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "TB", "RBI", 
                    "SB", "CS", "SH", "SF", "BB", "IBB", "HBP", "SO", "DP", "MHit"]

# 타자 데이터 그룹화 및 합계
cumulative_hitter_stats = tot_df_hitter.groupby(["Team", "Name", "No"])[sum_cols_hitter].sum().reset_index()

# 타자 비율 지표 재계산
cumulative_hitter_stats["AVG"] = (cumulative_hitter_stats["H"] / cumulative_hitter_stats["AB"]).round(3)
cumulative_hitter_stats["OBP"] = (
    (cumulative_hitter_stats["H"] + cumulative_hitter_stats["BB"] + cumulative_hitter_stats["HBP"]) /
    (cumulative_hitter_stats["AB"] + cumulative_hitter_stats["BB"] + cumulative_hitter_stats["HBP"] + cumulative_hitter_stats["SF"])
).round(3)
cumulative_hitter_stats["SLG"] = (cumulative_hitter_stats["TB"] / cumulative_hitter_stats["AB"]).round(3)
cumulative_hitter_stats["OPS"] = (cumulative_hitter_stats["OBP"] + cumulative_hitter_stats["SLG"]).round(3)
cumulative_hitter_stats["SB%"] = cumulative_hitter_stats["SB"] / (cumulative_hitter_stats["SB"] + cumulative_hitter_stats["CS"])
cumulative_hitter_stats["BB/K"] = cumulative_hitter_stats["BB"] / cumulative_hitter_stats["SO"]
cumulative_hitter_stats["XBH/H"] = (
    (cumulative_hitter_stats["2B"] + cumulative_hitter_stats["3B"] + cumulative_hitter_stats["HR"]) / cumulative_hitter_stats["H"]
)
# 투수 누적합 가능한 컬럼
sum_cols_pitcher = ["G", "W", "L", "SV", "HLD", "BF", "AB", "P", "IP", "HA", "HR", "SH", "SF",
                    "BB", "IBB", "HBP", "SO", "WP", "BK", "R", "ER"]
# 투수 데이터 그룹화 및 합계
cumulative_pitcher_stats = tot_df_pitcher.groupby(["Team", "Name", "No"])[sum_cols_pitcher].sum().reset_index()

# 잘못된 0.99 값을 올림 처리
cumulative_pitcher_stats.loc[np.isclose(cumulative_pitcher_stats['IP'] % 1, 0.99, atol=0.01), 'IP'] = np.ceil(cumulative_pitcher_stats['IP'])
# cumulative_pitcher_stats['IP'] = (cumulative_pitcher_stats['IP']+ 0.001).round(2) # 이닝수 계산 시 부동소수점 오차 해결
# 파생 변수 추가
# 방어율(ERA) 계산: (자책점 / 이닝) * 9 (예제로 자책점과 이닝 컬럼 필요)
if 'ER' in cumulative_pitcher_stats.columns and 'IP' in cumulative_pitcher_stats.columns:
    cumulative_pitcher_stats['ERA'] = ((cumulative_pitcher_stats['ER'] / cumulative_pitcher_stats['IP']) * 9).round(2)

# 이닝당 삼진/볼넷/피안타 계산 (예제로 삼진(K), 볼넷(BB), 피안타(HA) 컬럼 필요)
if 'SO' in cumulative_pitcher_stats.columns and 'BB' in cumulative_pitcher_stats.columns and 'HA' in df_pitcher.columns:
    cumulative_pitcher_stats['SO/IP'] = (cumulative_pitcher_stats['SO'] / cumulative_pitcher_stats['IP']).round(2)
    cumulative_pitcher_stats['BB/IP'] = (cumulative_pitcher_stats['BB'] / cumulative_pitcher_stats['IP']).round(2)
    cumulative_pitcher_stats['H/IP'] = (cumulative_pitcher_stats['HA'] / cumulative_pitcher_stats['IP']).round(2)
    cumulative_pitcher_stats['K9'] = (cumulative_pitcher_stats['SO/IP'] * 9)

# WHIP 계산: (볼넷 + 피안타) / 이닝
if 'BB' in cumulative_pitcher_stats.columns and 'HA' in cumulative_pitcher_stats.columns:
    cumulative_pitcher_stats['WHIP'] = ((cumulative_pitcher_stats['BB'] + cumulative_pitcher_stats['HA']) / cumulative_pitcher_stats['IP']).round(3)
    cumulative_pitcher_stats['BAA'] = (cumulative_pitcher_stats['HA'] / cumulative_pitcher_stats['AB']).round(3)
    cumulative_pitcher_stats['OBP'] = (cumulative_pitcher_stats['HA'] + cumulative_pitcher_stats['BB'] + cumulative_pitcher_stats['HBP']) / (cumulative_pitcher_stats['AB'] + cumulative_pitcher_stats['BB'] + cumulative_pitcher_stats['HBP'] + cumulative_pitcher_stats['SF']).round(3)

################################################################
## UI Tab
################################################################
## 탭 설정
tab_sn_players, tab_sn_teams, tab_sn_viz, tab_schd, tab_sn_league, tab_sn_terms, tab_dataload = st.tabs(["개인기록", "팀기록", "시각화/통계", "일정", 
                                                                                                          "전체선수", "약어", "데이터로딩"])
with tab_sn_players: # (팀별)개잉 선수기록 탭
    DATA_URL_B = "http://www.gameone.kr/club/info/ranking/hitter?club_idx={}&kind=&season={}".format(team_id, default_year)
    df_hitter_team = df_hitter.loc[df_hitter.Team == team_name].reset_index(drop=True).drop('Team', axis = 1)
    DATA_URL_P = "http://www.gameone.kr/club/info/ranking/pitcher?club_idx={}&kind=&season={}".format(team_id, default_year)
    df_pitcher_team = df_pitcher.loc[df_pitcher.Team == team_name].reset_index(drop=True).drop('Team', axis = 1)    

    # HTML display Setting
    span_stylesetting = '<span style="font-size: 11px; color: black; line-height: 5px;">'    
    if (df_hitter.shape[0] > 0) & (df_pitcher.shape[0] > 0) : # pitcher data exists    
        df_h_meandict = {k: round(v, 3) for k, v in df_hitter[rank_by_cols_h_sorted].mean(numeric_only=True).to_dict().items()}
        df_h_meandict_kr = {hitter_data_EnKr.get(k, k): v for k, v in df_h_meandict.items()}

        df_h_mediandict = {k: round(v, 3) for k, v in df_hitter[rank_by_cols_h_sorted].median(numeric_only=True).to_dict().items()}
        df_h_mediandict_kr = {hitter_data_EnKr.get(k, k): v for k, v in df_h_mediandict.items()}

        df_p_meandict = {k: round(v, 3) for k, v in df_pitcher[rank_by_cols_p_sorted].dropna().mean(numeric_only=True).to_dict().items()}
        df_p_meandict_kr = {pitcher_data_EnKr.get(k, k): v for k, v in df_p_meandict.items()}

        df_p_mediandict = {k: round(v, 3) for k, v in df_pitcher[rank_by_cols_p_sorted].dropna().median(numeric_only=True).to_dict().items()}
        df_p_mediandict_kr = {pitcher_data_EnKr.get(k, k): v for k, v in df_p_mediandict.items()}    

    tab_sn_players_h, tab_sn_players_p, tab_sn_players_ai = st.tabs(["타자 [{}명]".format(df_hitter_team.shape[0]), 
                                                                    "투수 [{}명]".format(df_pitcher_team.shape[0]),
                                                                    "AI 리포트"])

    with tab_sn_players_h: # 팀별 타자 탭
        if (df_hitter.shape[0] > 0) : # data exists            
            st.dataframe(df_hitter_team[['No', 'Name'] + rank_by_cols_h_sorted[1:]].sort_values(by = ['PA', 'AVG'], ascending = False).rename(columns = hitter_data_EnKr, inplace=False),
                        use_container_width = True, hide_index = True)
            st.write(DATA_URL_B)

            # 첫 번째 div 스타일
            h_box_stylesetting_1 = """
                <div style="
                    background-color: rgba(240, 240, 240, 0.8);  
                    color: #000000;                              
                    padding: 10px 12px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 11px;
                    margin-bottom: 10px;
                    border: 1px solid #ccc;
                ">
                <b>[전체 타자 평균값]</b><br>
                {}
                </div>
            """.format(", ".join([f"{k}: {v}" for k, v in df_h_meandict_kr.items()]))

            # 두 번째 div 스타일
            h_box_stylesetting_2 = """
                <div style="
                    background-color: rgba(240, 240, 240, 0.8);  
                    color: #000000;                              
                    padding: 10px 12px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 11px;
                    margin-bottom: 10px;
                    border: 1px solid #ccc;
                ">
                <b>[전체 타자 중앙값]</b><br>
                {}
                </div>
            """.format(", ".join([f"{k}: {v}" for k, v in df_h_mediandict_kr.items()]))
            with st.expander(f'{default_year}시즌 데이터셋 평균/중앙값(참고용)'):
                st.markdown(h_box_stylesetting_1 + " " + h_box_stylesetting_2, unsafe_allow_html=True)            

        filtered_cumulative_hitter_stats = cumulative_hitter_stats.loc[
            cumulative_hitter_stats['Team'] == team_name, 
            ['No', 'Name'] + rank_by_cols_h_sorted[1:]].sort_values(by = ['PA', 'AVG'], ascending = False).rename(columns = hitter_data_EnKr, inplace=False).reset_index(drop=True)
        
        st.write('')
        st.write(f'{team_name} : 타자 누적기록 [{len(filtered_cumulative_hitter_stats)}명]')
        st.dataframe(filtered_cumulative_hitter_stats, use_container_width = True, hide_index = True)

    with tab_sn_players_p: # 팀별 투수 탭
        if (df_pitcher.shape[0] > 0) :
            st.dataframe(df_pitcher_team[['No', 'Name'] + rank_by_cols_p_sorted[1:]].sort_values(by = ['IP', 'ERA'], ascending = False).rename(columns = pitcher_data_EnKr, inplace=False),
                        use_container_width = True, hide_index = True)
            st.write(DATA_URL_P)

            # 첫 번째 div 스타일
            p_box_stylesetting_1 = """
                <div style="
                    background-color: rgba(240, 240, 240, 0.8);  
                    color: #000000;                              
                    padding: 10px 12px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 11px;
                    margin-bottom: 10px;
                    border: 1px solid #ccc;
                ">
                <b>[전체 투수 평균값]</b><br>
                {}
                </div>
            """.format(", ".join([f"{k}: {v}" for k, v in df_p_meandict_kr.items()]))

            # 두 번째 div 스타일
            p_box_stylesetting_2 = """
                <div style="
                    background-color: rgba(240, 240, 240, 0.8);  
                    color: #000000;                              
                    padding: 10px 12px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 11px;
                    margin-bottom: 10px;
                    border: 1px solid #ccc;
                ">
                <b>[전체 투수 중앙값]</b><br>
                {}
                </div>
            """.format(", ".join([f"{k}: {v}" for k, v in df_p_mediandict_kr.items()]))
            with st.expander(f'{default_year}시즌 데이터셋 평균/중앙값(참고용)'):
                st.markdown(p_box_stylesetting_1 + " " + p_box_stylesetting_2, unsafe_allow_html=True)            

        filtered_cumulative_pitcher_stats = cumulative_pitcher_stats.loc[
            cumulative_pitcher_stats['Team'] == team_name, 
            ['No', 'Name'] + rank_by_cols_p_sorted[1:]].sort_values(by = ['IP', 'ERA'], ascending = False).rename(columns = pitcher_data_EnKr, inplace=False).reset_index(drop=True)
        
        st.write('')
        st.write(f'{team_name} : 투수 누적기록 [{len(filtered_cumulative_pitcher_stats)}명]')
        st.dataframe(filtered_cumulative_pitcher_stats, use_container_width = True, hide_index = True)

    with tab_sn_players_ai: # AI Report 탭
        st.write("본 리포트는 생성형 AI가 작성하였으므로, 구체적인 수치 및 사실관계는 확인이 필요합니다.")
        tab_sn_players_ai_topcol1, tab_sn_players_ai_topcol2 = st.columns([1, 1])
        with tab_sn_players_ai_topcol1:
            user_password_aireport = st.text_input('Input Password for AI Report', type='password', key='password_genai_h')
            user_password_aireport = str(user_password_aireport)
        with tab_sn_players_ai_topcol2:
            # 우선순위 모델
            priority_models = ['gemini-1.5-flash', 'gemini-2.0-flash']

            # 모델 리스트 가져오기 및 필터링
            available_models = genai.list_models()

            # 필터링
            filtered_models = []
            for model in available_models:
                name = model.name.split("/")[-1]
                methods = model.supported_generation_methods

                # 우선순위 모델이면 무조건 포함
                if name in priority_models:
                    filtered_models.append(name)
                    continue

                # 제외 조건: vision 포함, 멀티모달 지원, latest 없음
                if 'vision' in name.lower():
                    continue
                if 'generate_multimodal' in methods:
                    continue
                if 'latest' not in name.lower():
                    continue

                filtered_models.append(name)


            # 우선순위 모델 상단 배치
            model_list = [m for m in priority_models if m in filtered_models] + \
                        [m for m in filtered_models if m not in priority_models]

            # 선택 박스
            ai_model = st.selectbox('AI Model 선택', model_list, key='selbox_aimdl', index=0)

            # ai_model = st.selectbox('AI Model 선택', ['gemini-1.5-flash', 'gemini-2.5-pro-exp-03-25'], key = 'selbox_aimdl', index = 0)
        if user_password_aireport == st.secrets["password_gai"]: # Correct Password
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else st.text_input("🔑 Password", type="password")
            if GOOGLE_API_KEY:
                # Gemini 설정
                genai.configure(api_key=GOOGLE_API_KEY)
                try:
                    model = genai.GenerativeModel('models/{}'.format(ai_model))
                except :
                    model = genai.GenerativeModel("models/gemini-1.5-flash") #

                df_season = df_hitter_team[['No', 'Name'] + rank_by_cols_h_sorted[1:]].sort_values(by = ['PA', 'AVG'], ascending = False).rename(columns = hitter_data_EnKr, inplace=False) 
                df_total = filtered_cumulative_hitter_stats

                df_season_p = df_pitcher_team[['No', 'Name'] + rank_by_cols_p_sorted[1:]].sort_values(by = ['IP', 'ERA'], ascending = False).rename(columns = pitcher_data_EnKr, inplace=False)
                df_total_p = filtered_cumulative_pitcher_stats

                if (df_season is not None) & (df_season_p is not None):
                    # if st.button("🔍 Gemini AI Report"):
                    prompt_h = f"""
                    당신은 야구 데이터 분석가입니다. 이 데이터는 사회인야구의 특정 팀의 타자 데이터입니다. 해당팀의 데이터를 보고 이 팀에 대해 분석 보고서를 작성해야 하는 상황입니다.
                    이 데이터를 보고 이 팀에서 많은 타석을 소화한 타자를 우선적으로 고려하여 우수한 타격 성적을 나타내는 핵심선수를 3명정도 찾아주고, 해당 선수들의 특성을 분석해주세요.
                    데이터는 이번 시즌 이 팀의 타자 데이터와, 이번 시즌 리그 전체 팀의 중앙값, 그리고 통산 데이터로 구성되어 있습니다. 
                    특히 OPS로는 해당 타자의 공격력을, BB/K로는 해당 타자의 선구안을 평가할 수 있다고 생각합니다. 
                    그리고 사회인야구에서 홈런을 기록하는 것은 매우 어렵기 떄문에 통산 홈런이 있다면 해당 내용을 언급해주세요 
                    (특히 홈런 숫자를 언급할 떄는 틀리지 않도록 신중하게 생각하고 말해주세요! 자꾸 '3루타'랑 헷갈리는 것 같은데 혼동하지 않도록 주의).
                    이렇게 주는 이유는 이번 시즌 데이터를 분석할 때는 각 선수별 기록을 중앙값과 비교해 해당 선수의 수준을 정량적으로 비교/평가 하기 위함입니다.
                    이 데이터의 특성을 분석해 다음 내용을 포함하여 한국어로 간결하게 요약해 주십시오.
                    보고서 제목은 없이 바로 본론을 작성해주세요:

                        1. 주요 타자 이름(#배번) : 해당 선수의 특징적인 기록과, 중앙값 대비 각 선수들은 어떤 값을 갖고 있는지?(중앙값보다 큰지, 작은지?)
                        2. 간단한 해석 또는 인사이트 (3문장 이하)

                    데이터(시즌): {data_to_text(df_season)}
                    데이터(이번 시즌 전체 팀의 중앙값): {data_to_text(df_h_mediandict_kr)}
                    데이터(통산): {data_to_text(df_total)}
                    """
                    prompt_p = f"""
                    당신은 야구 데이터 분석가입니다. 이 데이터는 사회인야구의 특정 팀의 투수 데이터입니다. 해당팀의 데이터를 보고 이 팀에 대해 분석 보고서를 작성해야 하는 상황입니다.
                    이 데이터를 보고 이 팀에서 많은 이닝을 소화한 투수를 우선적으로 고려하여 우수한 기록을 나타내는 핵심선수를 3명정도 찾아주고, 해당 선수들의 특성을 분석해주세요.
                    데이터는 이번 시즌 이 팀의 투수 데이터와, 이번 시즌 리그 전체 팀의 중앙값, 그리고 통산 데이터로 구성되어 있습니다. 
                    특히 이닝당 삼진갯수로는 해당 투수의 구위를, 이닝당 볼넷갯수를 통해 해당 투수의 제구력을 평가할 수 있다고 생각합니다.
                    이렇게 주는 이유는 이번 시즌 데이터를 분석할 때는 각 선수별 기록을 중앙값과 비교해 해당 선수의 수준을 정량적으로 비교/평가 하기 위함입니다.
                    이 데이터의 특성을 분석해 다음 내용을 포함하여 한국어로 간결하게 요약해 주십시오.
                    보고서 제목은 없이 바로 본론을 작성해주세요.:

                        1. 주요 투수 이름(#배번) : 해당 선수의 특징적인 기록과, 중앙값 대비 각 선수들은 어떤 값을 갖고 있는지?(중앙값보다 큰지, 작은지?)
                        2. 간단한 해석 또는 인사이트 (3문장 이하)

                    데이터(시즌): {data_to_text(df_season_p)}
                    데이터(이번 시즌 전체 팀의 중앙값): {data_to_text(df_p_mediandict_kr)}
                    데이터(통산): {data_to_text(df_total_p)}
                    """
                    with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                        try:
                            response_h = model.generate_content(prompt_h)
                            response_p = model.generate_content(prompt_p)
                            tab_sn_players_ai_colh, tab_sn_players_ai_colp = st.columns(2)
                            with tab_sn_players_ai_colh:
                                # st.write("📈 Gemini AI 분석 결과 [타자]")
                                st.write(response_h.text)
                            with tab_sn_players_ai_colp:
                                # st.write("📈 Gemini AI 분석 결과 [투수]")
                                st.write(response_p.text)                                
                        except Exception as e:
                            st.error(f"Gemini API 호출 중 오류 발생: {e}")

        else:
            st.warning("비밀번호를 입력해주세요")

with tab_sn_teams: # 팀 기록 탭
    tab_sn_teams_allteams, tab_sn_teams_team = st.tabs(['전체 팀', '선택 팀 : {}'.format(team_name)])

    with tab_sn_teams_allteams: # 전체 팀 탭   
        # 공격지표 히트맵용 데이터프레임 준비
        hitter_heatmap_df = hitter_grpby_rank.copy()
        # 팀명 컬럼을 영어 팀명으로 매핑하여 'team_eng' 컬럼 생성
        hitter_heatmap_df['team_eng'] = hitter_heatmap_df['Team'].map(team_name_dict)
        # 팀명을 기준으로 우리팀이 맨위에 오도록 설정
        hitter_heatmap_df = hitter_heatmap_df.sort_values(by='Team')        # 1. Team 명 기준 오름차순 정렬
        target = hitter_heatmap_df[hitter_heatmap_df['Team'].str.contains(highlight_team)]  # 2. 특정 문자열이 있는 행 필터링
        others = hitter_heatmap_df[~hitter_heatmap_df['Team'].str.contains(highlight_team)]         # 3. 나머지 행 필터링
        hitter_heatmap_df = pd.concat([target, others], ignore_index=True)  # 4. 두 데이터프레임을 위에서 아래로 concat
        
        # 기존 'Team' 컬럼 제거 후 'team_eng'를 인덱스로 설정
        hitter_heatmap_df = hitter_heatmap_df.drop(['Team', 'PA', 'AB'], axis=1).copy()
        hitter_heatmap_df.set_index('team_eng', inplace=True)
    
        # 수비지표 히트맵용 데이터프레임 준비
        pitcher_heatmap_df = pitcher_grpby_rank.copy()
        # 팀명을 영어로 매핑하여 'team_eng' 컬럼 생성
        pitcher_heatmap_df['team_eng'] = pitcher_heatmap_df['Team'].map(team_name_dict)
        # 팀명을 기준으로 우리팀이 맨위에 오도록 설정
        pitcher_heatmap_df = pitcher_heatmap_df.sort_values(by='Team')        # 1. Team 명 기준 오름차순 정렬
        target = pitcher_heatmap_df[pitcher_heatmap_df['Team'].str.contains(highlight_team)]  # 2. 특정 문자열이 있는 행 필터링
        others = pitcher_heatmap_df[~pitcher_heatmap_df['Team'].str.contains(highlight_team)]         # 3. 나머지 행 필터링
        pitcher_heatmap_df = pd.concat([target, others], ignore_index=True)  # 4. 두 데이터프레임을 위에서 아래로 concat

        # 기존 'Team' 컬럼 제거 후 'team_eng'를 인덱스로 설정
        pitcher_heatmap_df = pitcher_heatmap_df.drop(['Team', 'BF', 'AB'], axis=1).copy()
        pitcher_heatmap_df.set_index('team_eng', inplace=True)

        # 커스텀 컬러맵 설정 (어두운 빨강 → 흰색)
        colors = ["#8b0000", "#ffffff"]
        cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=15)

        tab_sn_teams_allteams_heatmap_left, tab_sn_teams_allteams_heatmap_right = st.columns(2)

        with tab_sn_teams_allteams_heatmap_left:
            # 공격지표 히트맵 생성 및 출력
            fig_hitter = create_heatmap(hitter_heatmap_df, cmap, input_figsize=(10, 5))
            st.pyplot(fig_hitter)
            # plt.close(fig_hitter)  # 필요시 리소스 해제

        with tab_sn_teams_allteams_heatmap_right:
            # 수비지표 히트맵 생성 및 출력
            fig_pitcher = create_heatmap(pitcher_heatmap_df, cmap, input_figsize=(10, 5))
            st.pyplot(fig_pitcher)
            # plt.close(fig_pitcher)  # 히트맵 리소스 해제

        # 공격지표 테이블 출력
        st.write('공격지표')
        st.dataframe(
            hitter_grpby.loc[:, rank_by_cols_h_sorted].rename(columns=hitter_data_EnKr, inplace=False),
            use_container_width=True,
            hide_index=True
        )
        # 공격지표 순위 테이블 확장 영역
        with st.expander('공격지표 순위 테이블'):
            st.dataframe(
                hitter_grpby_rank.rename(columns=hitter_data_EnKr, inplace=False),
                use_container_width=True,
                hide_index=True
            )

        ############################################################
        # 수비지표 테이블 출력
        st.write('수비지표')
        st.dataframe(
            pitcher_grpby.loc[:, rank_by_cols_p_sorted].rename(columns=pitcher_data_EnKr, inplace=False),
            use_container_width=True,
            hide_index=True
        )
        
        # 수비지표 순위 테이블 확장 영역
        with st.expander('수비지표 순위 테이블'):
            st.dataframe(
                pitcher_grpby_rank.rename(columns=pitcher_data_EnKr, inplace=False),
                use_container_width=True,
                hide_index=True
            )

    with tab_sn_teams_team: # 선택 팀 기록 탭
        # 메인팀 공격/수비지표 따로 필터링해 변수에 저장
        mainteam_name = highlight_team 
        df1_h = hitter_grpby.loc[hitter_grpby.Team == mainteam_name, rank_by_cols_h_sorted].drop('Team', axis = 1) # , use_container_width = True, hide_index = True)
        df2_h = hitter_grpby_rank.loc[hitter_grpby_rank.Team == mainteam_name].drop('Team', axis = 1)
        df1_h.insert(0, '공격지표', '기록')
        df2_h.insert(0, '공격지표', '순위')
        mainteam_statrank_h = pd.concat([df1_h, df2_h], axis = 0).rename(columns = hitter_data_EnKr, inplace=False).set_index('공격지표').reset_index()
        
        # 선택한 팀의 팀 수비지표 출력
        df1_p = pitcher_grpby.loc[pitcher_grpby.Team == mainteam_name, rank_by_cols_p_sorted].drop('Team', axis = 1)
        df2_p = pitcher_grpby_rank.loc[pitcher_grpby_rank.Team == mainteam_name].drop('Team', axis = 1)
        df1_p.insert(0, '수비지표', '기록')
        df2_p.insert(0, '수비지표', '순위')
        mainteam_statrank_p = pd.concat([df1_p, df2_p], axis = 0).rename(columns = pitcher_data_EnKr, inplace=False).set_index('수비지표').reset_index()


        tab_sn_teams_team_col1, tab_sn_teams_team_col2 = st.columns(2)
        ############################################################
        with tab_sn_teams_team_col1:
            # 선택한 팀의 팀 공격지표 출력
            df1_h = hitter_grpby.loc[hitter_grpby.Team == team_name, rank_by_cols_h_sorted].drop('Team', axis = 1) # , use_container_width = True, hide_index = True)
            df2_h = hitter_grpby_rank.loc[hitter_grpby_rank.Team == team_name].drop('Team', axis = 1)
            df1_h.insert(0, '공격지표', '기록')
            df2_h.insert(0, '공격지표', '순위')
            team_statrank_h = pd.concat([df1_h, df2_h], axis = 0).rename(columns = hitter_data_EnKr, inplace=False).set_index('공격지표')
            team_statrank_h_html_table = team_statrank_h.T.to_html(formatters=[format_cell] * team_statrank_h.T.shape[1], escape=False) 
            # 최종 HTML 조합
            st.components.v1.html(table_style_12px + apply_row_styling(team_statrank_h_html_table), 
                                  height=750, scrolling=True)   
            if team_name != mainteam_name : # 사용자 입력팀이 메인팀이 아닐때 
                # 두 번째 div 스타일
                mainteam_box_stylesetting = """
                    <div style="
                        background-color: rgba(240, 240, 240, 0.8);  
                        color: #000000;                              
                        padding: 10px 12px;
                        border-radius: 8px;
                        font-family: monospace;
                        font-size: 11px;
                        margin-bottom: 10px;
                        border: 1px solid #ccc;
                    ">
                    <b>[{}]</b><br>
                    {}
                    </div>
                """.format(
                    mainteam_name, 
                    ", ".join([f"{k}: {v[0]} [{int(v[1])}위]" for k, v in list(mainteam_statrank_h.to_dict().items())[1:]])
                                #mainteam_statrank_h.to_dict().items()])
                )
                st.markdown(mainteam_box_stylesetting, unsafe_allow_html=True)

        ############################################################
        with tab_sn_teams_team_col2:
            # 선택한 팀의 팀 수비지표 출력
            df1_p = pitcher_grpby.loc[pitcher_grpby.Team == team_name, rank_by_cols_p_sorted].drop('Team', axis = 1)
            df2_p = pitcher_grpby_rank.loc[pitcher_grpby_rank.Team == team_name].drop('Team', axis = 1)
            df1_p.insert(0, '수비지표', '기록')
            df2_p.insert(0, '수비지표', '순위')
            team_statrank_p = pd.concat([df1_p, df2_p], axis = 0).rename(columns = pitcher_data_EnKr, inplace=False).set_index('수비지표')
            # st.dataframe(team_statrank_p.T) #, use_container_width = True, hide_index = True)   
            team_statrank_p_html_table = team_statrank_p.T.to_html(formatters=[format_cell] * team_statrank_p.T.shape[1], escape=False) 
            # .to_html(classes='table table-striped', border=0)
            # Streamlit에서 HTML 출력
            # st.markdown(team_statrank_p_html_table, unsafe_allow_html=True)
            # 최종 HTML 조합
            st.components.v1.html(table_style_12px + apply_row_styling(team_statrank_p_html_table), 
                                  height=750, scrolling=True)
            if team_name != mainteam_name : # 사용자 입력팀이 메인팀이 아닐때 
                st.write(mainteam_statrank_p)
                mainteam_box_stylesetting_p = """
                    <div style="
                        background-color: rgba(240, 240, 240, 0.8);  
                        color: #000000;                              
                        padding: 10px 12px;
                        border-radius: 8px;
                        font-family: monospace;
                        font-size: 11px;
                        margin-bottom: 10px;
                        border: 1px solid #ccc;
                    ">
                    <b>[{}]</b><br>
                    {}
                    </div>
                """.format(
                    mainteam_name, 
                    ", ".join([f"{k}: {v[0]} [{int(v[1])}위]" for k, v in list(mainteam_statrank_p.to_dict().items())[1:]])
                                #mainteam_statrank_h.to_dict().items()])
                )
                st.markdown(mainteam_box_stylesetting_p, unsafe_allow_html=True)

with tab_sn_viz:
    tab_sn_viz_1, tab_sn_viz_2, tab_sn_viz_3 = st.tabs(["선수별분포", "팀별비교", "통계량"])
    with tab_sn_viz_1: # 개인 선수별 기록 분포 시각화
        #st.subheader('선수별 기록 분포 시각화')    
        df_plot = df_hitter
        tab_sn_viz_col1, tab_sn_viz_col2, tab_sn_viz_col3 = st.columns(3)
        with tab_sn_viz_col1:        # 데이터셋 선택을 위한 토글 버튼
            dataset_choice = st.radio('데이터셋 선택', ('타자', '투수'), key = 'dataset_choice')
        with tab_sn_viz_col2:         # 그래프 유형 선택을 위한 토글 버튼
            graph_type = st.radio('그래프 유형', ('히스토그램', '박스플롯'), key = 'graph_type')
        with tab_sn_viz_col3:
            colsNo = st.selectbox('한 줄에 몇개 표시할까요? (1~4열):', options=[1, 2, 3, 4], index=2)

        # 선택된 데이터셋에 따라 데이터 프레임 설정
        if dataset_choice == '투수':
            df_plot = df_pitcher.copy()
        else:
            df_plot = df_hitter.copy()

        numeric_columns = df_plot.select_dtypes(include=['float', 'int']).columns
        rows = (len(numeric_columns) + colsNo - 1) // colsNo
        fig, axs = plt.subplots(rows, colsNo, figsize=(15, 3 * rows))

        # axs가 1차원 배열일 경우 처리
        if rows * colsNo == 1:
            axs = [axs]
        elif rows == 1 or colsNo == 1:
            axs = axs.flatten()
        else:
            axs = axs.reshape(-1)

        # "Plotting" 버튼 추가
        if st.button('Plotting', key = 'dist_btn'):
            for i, var in enumerate(numeric_columns):
                ax = axs[i]
                if graph_type == '히스토그램':
                    sns.histplot(df_plot[var].dropna(), kde=False, ax=ax)
                    ax.set_title(f'{var}')
                elif graph_type == '박스플롯':
                    sns.boxplot(x=df_plot[var].dropna(), ax=ax)
                    ax.set_title(f'{var}')

            # 빈 서브플롯 숨기기
            for j in range(len(numeric_columns), rows * colsNo):
                axs[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

    ### template_input 
    # plotly - Plotly의 기본 템플릿.     # plotly_white - 배경이 하얀색인 깔끔한 템플릿.     # plotly_dark - 배경이 어두운색인 템플릿.
    # ggplot2 - R의 ggplot2 스타일을 모방한 템플릿.    # seaborn - Python의 seaborn 라이브러리 스타일을 모방한 템플릿.    # simple_white - 매우 단순하고 깨끗한 템플릿.
    
    with tab_sn_viz_2: # tab_sn_vs [레이더 차트]
        teams = list(sorted(team_id_dict.keys())) # Team list applied sorting
        template_input = 'plotly_white'    
        try:
            # '호시탐탐'의 인덱스 찾기
            idx_hstt = teams.index('코메츠 호시탐탐')
        except ValueError:
            idx_hstt = 0

        # st.subheader('팀 간 전력 비교')      
        tab_sn_vs_col1, tab_sn_vs_col2, tab_sn_vs_col3 = st.columns(3)
        with tab_sn_vs_col1:        # 2개 팀을 비교할지 / 전체 팀을 한판에 그릴지 선택하는 토글 버튼
            team_all = st.toggle("Select All Teams")
        with tab_sn_vs_col2:         # # 스트림릿 셀렉트박스로 팀 선택
            if not team_all: #team_selection_rader == 'VS':            # 스트림릿 셀렉트박스로 팀 선택
                team1 = st.selectbox('Select Team 1:', options = teams, index=idx_hstt)
        with tab_sn_vs_col3:  
            if not team_all: #if team_selection_rader == 'VS':            # 스트림릿 셀렉트박스로 팀 선택              
                team2 = st.selectbox('Select Team 2:', options = teams, index=1)
        multisel_h = st.multiselect('공격(타자) 지표 선택',
            [hitter_data_EnKr.get(col, col) for col in rank_by_cols_h_sorted[1:]], 
            ['타율', '출루율', 'OPS', '볼넷', '삼진', '도루'], max_selections = 12
        )
        multisel_p = st.multiselect('수비(투수) 지표 선택',
            # rank_by_cols_p_sorted, 
            [pitcher_data_EnKr.get(col, col) for col in rank_by_cols_p_sorted[1:]],
            ['방어율', 'WHIP', 'H/IP', 'BB/IP', 'SO/IP', '피안타율'], max_selections = 12
        )        
        # "Plotting" 버튼 추가
        if st.button('Plotting', key = 'vs_rader_btn'):
            hitter_grpby_plt = hitter_grpby.rename(columns = hitter_data_EnKr, inplace=False).copy()
            pitcher_grpby_plt = pitcher_grpby.rename(columns = pitcher_data_EnKr, inplace=False) .copy()
            selected_cols_h = ['팀'] + multisel_h # ['AVG', 'OBP', 'OPS', 'BB', 'SO', 'SB']
            selected_cols_p = ['팀'] + multisel_p
            # 데이터 스케일링
            hitter_grpby_plt_scaled = hitter_grpby_plt.rename(columns = hitter_data_EnKr, inplace=False).copy()
            scaler_h = MinMaxScaler()             # 스케일러 초기화
            hitter_grpby_plt_scaled[hitter_grpby_plt_scaled.columns[1:]] = scaler_h.fit_transform(hitter_grpby_plt_scaled.iloc[:, 1:]) # 첫 번째 열 'Team'을 제외하고 스케일링
            pitcher_grpby_plt_scaled = pitcher_grpby_plt.rename(columns = pitcher_data_EnKr, inplace=False).copy()
            scaler_p = MinMaxScaler()             # 스케일러 초기화
            pitcher_grpby_plt_scaled[pitcher_grpby_plt_scaled.columns[1:]] = scaler_p.fit_transform(pitcher_grpby_plt_scaled.iloc[:, 1:]) # 첫 번째 열 'Team'을 제외하고 스케일링
            if team_all: #if team_selection_rader == '전체':
                filtered_data_h = hitter_grpby_plt_scaled
                radar_data_h = filtered_data_h[selected_cols_h].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                fig_h = px.line_polar(radar_data_h, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'공격력')   

                filtered_data_p = pitcher_grpby_plt_scaled
                radar_data_p = filtered_data_p[selected_cols_p].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                fig_p = px.line_polar(radar_data_p, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'수비력')  

            else: # team_selection_rader == 'VS' : 2개팀을 비교할 경우
                # 선택된 팀 데이터 필터링
                filtered_data_h = hitter_grpby_plt_scaled[hitter_grpby_plt_scaled['팀'].isin([team1, team2])]#.rename(columns = hitter_data_EnKr, inplace=False).copy()
                # 레이더 차트 데이터 준비
                radar_data_h = filtered_data_h[selected_cols_h].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                # 레이더 차트 생성
                fig_h = px.line_polar(radar_data_h, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'공격력 : {team1} vs {team2}')
                # 선택된 팀 데이터 필터링
                filtered_data_p = pitcher_grpby_plt_scaled[pitcher_grpby_plt_scaled['팀'].isin([team1, team2])]#.rename(columns = pitcher_data_EnKr, inplace=False).copy()
                # 레이더 차트 데이터 준비
                radar_data_p = filtered_data_p[selected_cols_p].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                # 레이더 차트 생성
                fig_p = px.line_polar(radar_data_p, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'수비력 : {team1} vs {team2}')
            ########################
            ## Chart AND Dataframe display Area
            if not team_all:    #if team_selection_rader == 'VS':  
                df_rader_vs_h = pd.concat([hitter_grpby_plt.loc[hitter_grpby_plt['팀'] == team1, selected_cols_h], 
                                    hitter_grpby_plt.loc[hitter_grpby_plt['팀'] == team2, selected_cols_h]], axis = 0).sort_values('팀')      
                st.dataframe(df_rader_vs_h, use_container_width = True, hide_index = True) 
            else :
                st.dataframe(hitter_grpby_plt[selected_cols_h].sort_values('팀').T, use_container_width = True)    

            if not team_all:    #if team_selection_rader == 'VS':    
                df_rader_vs_p = pd.concat([pitcher_grpby_plt.loc[pitcher_grpby_plt['팀'] == team1, selected_cols_p], 
                                    pitcher_grpby_plt.loc[pitcher_grpby_plt['팀'] == team2, selected_cols_p]], axis = 0).sort_values('팀')           
                st.dataframe(df_rader_vs_p, use_container_width = True, hide_index = True)      
            else :
                st.dataframe(pitcher_grpby_plt[selected_cols_p].sort_values('팀').T, use_container_width = True)  

            tab_sn_vs_col2_1, tab_sn_vs_col2_2 = st.columns(2)   
            with tab_sn_vs_col2_1:            # 차트 보기 [Hitter]
                st.plotly_chart(fig_h, use_container_width=True)
            with tab_sn_vs_col2_2:             # 차트 보기 [Pitcher]
                st.plotly_chart(fig_p, use_container_width=True)
    with tab_sn_viz_3:
        st.write("선수 별 기록 분포 통계량")
        st.write("타자")
        st.dataframe(df_hitter.drop('No', axis = 1).rename(columns = hitter_data_EnKr, inplace=False).describe(), 
                     use_container_width = True, hide_index = False)  
        st.write("투수")
        st.dataframe(df_pitcher.drop('No', axis = 1).rename(columns = pitcher_data_EnKr, inplace=False).describe(), 
                     use_container_width = True, hide_index = False)  

with tab_schd:
    st.markdown(soup.find('span', {'class': 'info'}), unsafe_allow_html=True) # 시즌 기록 출력
    st.write('') # 한줄 공백
    # 강조할 팀명에 서식 적용
    highlighted_team = f"<span style='font-weight: bold; color: navy;'>{highlight_team}</span>" 
        #f"<b>{highlight_team}</b>"

    # 인덱스 없이 HTML 테이블로 출력
    df_schd2 = df_schd2[['일시', '구장', '선공', '선', '후', '후공', '결과']]
    schd_html_str = df_schd2.to_html(index=False, escape=False)
    # '코메츠 호시탐탐' 강조 처리
    schd_html_str = schd_html_str.replace(highlight_team, highlighted_team)
    # 최종 HTML 조합
    st.components.v1.html(table_style + apply_row_styling(schd_html_str), height=500, scrolling=True)
    st.write(schd_url)    

with tab_sn_league: # 전체 선수들의 기록을 출력해주는 탭
    tab_sn_players_1, tab_sn_players_2 = st.tabs(['타자 [{}명]'.format(df_hitter.shape[0]), '투수 [{}명]'.format(df_pitcher.shape[0])])
    with tab_sn_players_1: # 전체 선수 탭 > "성남:전체타자" 탭
        st.dataframe(df_hitter[['No', 'Name'] + rank_by_cols_h_sorted].rename(columns = hitter_data_EnKr, inplace=False), 
                     use_container_width = True, hide_index = True)

    with tab_sn_players_2: # 전체 선수 탭 > "성남:전체투수" 탭
        if df_pitcher.shape[0] > 0 : # pitcher data exists
            st.dataframe(df_pitcher[p_existing_columns].rename(columns = pitcher_data_EnKr, inplace=False), use_container_width = True, hide_index = True)
     
with tab_sn_terms: # 약어 설명
    # st.subheader('야구 기록 설명')
    hitters_term_table_html = """
        <div class="table-box">
            <table>
                <tr><th>ENG</th><th>KOR</th><th>Desc</th></tr>
                <tr><td>Name</td><td>성명</td><td>Player's name</td></tr>
                <tr><td>No</td><td>배번</td><td>Jersey number</td></tr>
                <tr><td>AVG</td><td>타율</td><td>Batting average</td></tr>
                <tr><td>G</td><td>경기</td><td>Games played</td></tr>
                <tr><td>PA</td><td>타석</td><td>Plate appearances</td></tr>
                <tr><td>AB</td><td>타수</td><td>At bats</td></tr>
                <tr><td>R</td><td>득점</td><td>Runs</td></tr>
                <tr><td>H</td><td>총안타</td><td>Hits</td></tr>
                <tr><td>1B</td><td>1루타</td><td>Singles</td></tr>
                <tr><td>2B</td><td>2루타</td><td>Doubles</td></tr>
                <tr><td>3B</td><td>3루타</td><td>Triples</td></tr>
                <tr><td>HR</td><td>홈런</td><td>Home runs</td></tr>
                <tr><td>TB</td><td>루타</td><td>Total bases</td></tr>
                <tr><td>RBI</td><td>타점</td><td>Runs batted in</td></tr>
                <tr><td>SB</td><td>도루</td><td>Stolen bases</td></tr>
                <tr><td>CS</td><td>도실</td><td>Caught stealing</td></tr>
                <tr><td>SH</td><td>희타</td><td>Sacrifice hits</td></tr>
                <tr><td>SF</td><td>희비</td><td>Sacrifice flies</td></tr>
                <tr><td>BB</td><td>볼넷</td><td>Walks</td></tr>
                <tr><td>IBB</td><td>고의4구</td><td>Intentional walks</td></tr>
                <tr><td>HBP</td><td>사구</td><td>Hit by pitch</td></tr>
                <tr><td>SO</td><td>삼진</td><td>Strikeouts</td></tr>
                <tr><td>DP</td><td>병살</td><td>Double plays</td></tr>
                <tr><td>SLG</td><td>장타율</td><td>Slugging percentage</td></tr>
                <tr><td>OBP</td><td>출루율</td><td>On-base percentage</td></tr>
                <tr><td>SB%</td><td>도루성공률</td><td>Stolen base %</td></tr>
                <tr><td>MHit</td><td>멀티히트</td><td>Multi-hit games</td></tr>
                <tr><td>OPS</td><td>OPS</td><td>On-base plus slugging</td></tr>
                <tr><td>BB/K</td><td>BB/K</td><td>Walks per strikeout</td></tr>
                <tr><td>XBH/H</td><td>장타/안타</td><td>Extra base hits per hit</td></tr>
                <tr><td>Team</td><td>팀</td><td>Team name</td></tr>
            </table>
        </div>
    """

    pitchers_term_table_html = """
        <div class="table-box">
            <table>
                <tr><th>ENG</th><th>KOR</th><th>Desc</th></tr>
                <tr><td>Name</td><td>성명</td><td>Player's name</td></tr>
                <tr><td>No</td><td>배번</td><td>Jersey number</td></tr>
                <tr><td>ERA</td><td>방어율</td><td>Earned run average</td></tr>
                <tr><td>WHIP</td><td>WHIP</td><td>Walks plus hits per inning</td></tr>
                <tr><td>SO/IP</td><td>이닝 당 탈삼진</td><td>Strikeouts per inning</td></tr>
                <tr><td>GS</td><td>경기수</td><td>Games started</td></tr>
                <tr><td>W</td><td>승</td><td>Wins</td></tr>
                <tr><td>L</td><td>패</td><td>Losses</td></tr>
                <tr><td>SV</td><td>세</td><td>Saves</td></tr>
                <tr><td>HLD</td><td>홀드</td><td>Holds</td></tr>
                <tr><td>BF</td><td>타자</td><td>Batters faced</td></tr>
                <tr><td>AB</td><td>타수</td><td>At bats against</td></tr>
                <tr><td>P</td><td>투구수</td><td>Pitches thrown</td></tr>
                <tr><td>HA</td><td>피안타</td><td>Hits allowed</td></tr>
                <tr><td>HR</td><td>피홈런</td><td>Home runs allowed</td></tr>
                <tr><td>SH</td><td>희생타</td><td>Sacrifice hits allowed</td></tr>
                <tr><td>SF</td><td>희생플라이</td><td>Sacrifice flies allowed</td></tr>
                <tr><td>BB</td><td>볼넷</td><td>Walks allowed</td></tr>
                <tr><td>IBB</td><td>고의4구</td><td>Intentional walks allowed</td></tr>
                <tr><td>HBP</td><td>사구</td><td>Hit by pitch allowed</td></tr>
                <tr><td>SO</td><td>탈삼진</td><td>Strikeouts</td></tr>
                <tr><td>WP</td><td>폭투</td><td>Wild pitches</td></tr>
                <tr><td>BK</td><td>보크</td><td>Balks</td></tr>
                <tr><td>R</td><td>실점</td><td>Runs allowed</td></tr>
                <tr><td>ER</td><td>자책점</td><td>Earned runs allowed</td></tr>
                <tr><td>IP</td><td>이닝</td><td>Innings pitched</td></tr>
            </table>
        </div>
    """

    tab_sn_terms_col1, tab_sn_terms_col2 = st.columns(2)
    # 스트림릿 페이지 제목 설정
    with tab_sn_terms_col1:        # 타자 데이터 설명
        st.components.v1.html(table_style + apply_row_styling(hitters_term_table_html), height=800, scrolling=True)
    with tab_sn_terms_col2:        # 투수 데이터 설명
        st.components.v1.html(table_style + apply_row_styling(pitchers_term_table_html), height=800, scrolling=True)

with tab_dataload:
    user_password_update = st.text_input('Input Password for Update', type='password')
    user_password_update = str(user_password_update)
    if user_password_update == st.secrets["password_update"]: # Correct Password
        st.write('Correct Password')
        dataload_year = st.selectbox('데이터 수집 년도', year_list, index = 0, key = 'dataload_year_selectbox')
        st.write('아래 버튼을 누르면 현재 시점의 데이터를 새로 로드합니다.')        
        if st.button('Data Update'):
            hitters = []
            pitchers = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(load_data, team_name, team_id, dataload_year): team_name for team_name, team_id in team_id_dict.items()}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        hitters.append(result['hitter'])
                        pitchers.append(result['pitcher'])
                    except Exception as exc:
                        print(f'Team {futures[future]} generated an exception: {exc}')
            # 모든 데이터를 각각의 데이터프레임으로 합침
            final_hitters_data = pd.concat(hitters, ignore_index=True)
            final_pitchers_data = pd.concat(pitchers, ignore_index=True)

            # 데이터프레임 df의 컬럼 자료형 설정
            df_hitter = final_hitters_data.astype(hitter_data_types)
            # 타자 데이터프레임 컬럼명 영어로
            df_hitter.columns = ['Name', 'No', 'AVG', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 
                                'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'SLG', 'OBP', 'SB%', 'MHit', 
                                'OPS', 'BB/K', 'XBH/H', 'Team']

            final_pitchers_data.loc[final_pitchers_data.방어율 == '-', '방어율'] = np.nan

            # 투수 데이터프레임 df_pitcher의 컬럼 자료형 설정
            df_pitcher = final_pitchers_data.astype(pitcher_data_types)
            # 투수 데이터프레임 컬럼명 영어로
            df_pitcher.columns = ['Name', 'No', 'ERA', 'G', 'W', 'L', 'SV', 'HLD', 'WPCT', 
                                'BF', 'AB', 'P', 'IP', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'WP', 'BK', 
                                'R', 'ER', 'WHIP', 'BAA', 'K9', 'Team']
            # IP 컬럼을 올바른 소수 형태로 변환
            df_pitcher['IP'] = df_pitcher['IP'].apply(lambda x: int(x) + (x % 1) * 10 / 3).round(2)
            
            ###### GOOGLE SHEETS
            # Create GSheets connection
            conn = st.connection("gsheets", type=GSheetsConnection)

            df_hitter = conn.update(worksheet="df_hitter_{}".format(dataload_year), data=df_hitter)
            df_pitcher = conn.update(worksheet="df_pitcher_{}".format(dataload_year), data=df_pitcher)
            time.sleep(3)
            st.toast('Saved Data from Web to Cloud! (Updated)', icon='☁️')
            st.write(df_hitter.shape, "Hitter Data SAVED!")
            st.dataframe(df_hitter, use_container_width = True, hide_index = True)
            st.write(df_pitcher.shape, "Pitcher Data SAVED!")
            st.dataframe(df_pitcher, use_container_width = True, hide_index = True)
    else:
        st.write('Wrong Password!!')

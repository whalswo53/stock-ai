"""
Trusted news sources, ticker name maps, and filter keywords.
"""

# News source whitelist by market — used by NewsCollector to filter articles
TRUSTED_PUBLISHERS: dict[str, list[str]] = {
    "KOSPI": [
        "한국경제", "한경", "매일경제", "매경", "이데일리",
        "연합뉴스", "뉴스1", "조선비즈", "서울경제",
        "파이낸셜뉴스", "머니투데이", "헤럴드경제", "아시아경제",
    ],
    "KOSDAQ": [
        "한국경제", "한경", "매일경제", "매경", "이데일리",
        "연합뉴스", "뉴스1", "조선비즈", "서울경제",
        "파이낸셜뉴스", "머니투데이", "헤럴드경제", "아시아경제",
    ],
    "NASDAQ": [
        "Reuters", "Bloomberg", "Associated Press", "AP",
        "MarketWatch", "CNBC", "Wall Street Journal", "WSJ",
        "Financial Times", "FT", "Seeking Alpha", "Barron's",
        "Motley Fool", "Benzinga", "Investopedia",
    ],
    "NYSE": [
        "Reuters", "Bloomberg", "Associated Press", "AP",
        "MarketWatch", "CNBC", "Wall Street Journal", "WSJ",
        "Financial Times", "FT", "Seeking Alpha", "Barron's",
        "Motley Fool", "Benzinga", "Investopedia",
    ],
}

# RSS feed URLs by market (feedparser secondary source)
RSS_FEEDS: dict[str, list[str]] = {
    "KOSPI": [
        "https://www.hankyung.com/feed/economy",
        "https://www.mk.co.kr/rss/50400012/",
        "https://www.edaily.co.kr/rss/feeds/edaily.xml",
        "https://www.yna.co.kr/RSS/economy.xml",
    ],
    "KOSDAQ": [
        "https://www.hankyung.com/feed/economy",
        "https://www.mk.co.kr/rss/50400012/",
        "https://www.edaily.co.kr/rss/feeds/edaily.xml",
    ],
    "NASDAQ": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    ],
    "NYSE": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.marketwatch.com/rss/topstories",
    ],
}

# Keywords that indicate low-quality or spam content — articles containing
# these are discarded before sentiment analysis
BLOCKED_KEYWORDS: set[str] = {
    "광고", "홍보", "PR", "sponsored", "advertisement",
    "클릭", "무료", "이벤트", "경품", "쿠폰",
    "클릭하면", "바로가기", "지금바로",
}

# ── 섹터 키워드 (뉴스 관련성 점수용) ──────────────────────────────────────────
# 종목명이 직접 언급되지 않아도 섹터 키워드가 잡히면 "간접 관련" 기사로
# 채택하되 관련도 점수를 낮게 부여한다 (NewsCollector.score_relevance).
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "반도체": [
        "반도체", "파운드리", "HBM", "D램", "DRAM", "낸드", "NAND",
        "웨이퍼", "칩", "메모리", "시스템반도체", "팹리스", "EUV",
        "semiconductor", "foundry", "chip", "memory", "wafer",
    ],
    "2차전지": [
        "2차전지", "이차전지", "배터리", "양극재", "음극재", "전고체",
        "리튬", "LFP", "NCM", "전기차 배터리",
        "battery", "lithium", "cathode", "EV battery",
    ],
    "플랫폼": [
        "플랫폼", "포털", "검색엔진", "메신저", "웹툰", "커머스",
        "platform", "e-commerce", "social media",
    ],
    "AI·빅테크": [
        "인공지능", "AI", "생성형", "LLM", "데이터센터", "클라우드",
        "GPU", "빅테크", "artificial intelligence", "data center",
        "cloud", "generative",
    ],
    "금융": [
        "은행", "금융지주", "기준금리", "대출", "예금", "증권",
        "금리", "banking", "interest rate", "loan",
    ],
    "자동차": [
        "자동차", "전기차", "완성차", "자율주행", "모빌리티",
        "EV", "autonomous", "automotive",
    ],
    "바이오": [
        "바이오", "제약", "신약", "임상", "FDA", "바이오시밀러",
        "biotech", "pharma", "clinical trial",
    ],
    "방산·에너지": [
        "방산", "방위산업", "원전", "원자력", "에너지",
        "defense", "nuclear",
    ],
}

# 티커 → 섹터 키워드 그룹 (SECTOR_KEYWORDS 키). 여기 없는 티커는
# 섹터 키워드 매칭 없이 직접 언급 필터만 적용된다.
TICKER_SECTOR: dict[str, str] = {
    # 반도체
    "005930.KS": "반도체", "000660.KS": "반도체", "042700.KS": "반도체",
    "000990.KS": "반도체", "058470.KQ": "반도체", "039030.KQ": "반도체",
    "NVDA": "반도체", "AMD": "반도체", "INTC": "반도체", "QCOM": "반도체",
    "AVGO": "반도체", "TSM": "반도체", "UMC": "반도체", "GFS": "반도체",
    "MU": "반도체", "ARM": "반도체", "0981.HK": "반도체", "1347.HK": "반도체",
    # 2차전지
    "373220.KS": "2차전지", "006400.KS": "2차전지", "051910.KS": "2차전지",
    "247540.KQ": "2차전지", "086520.KQ": "2차전지", "003670.KS": "2차전지",
    # 플랫폼
    "035720.KS": "플랫폼", "035420.KS": "플랫폼",
    # AI·빅테크
    "AAPL": "AI·빅테크", "MSFT": "AI·빅테크", "GOOGL": "AI·빅테크",
    "AMZN": "AI·빅테크", "META": "AI·빅테크", "NFLX": "AI·빅테크",
    "CRM": "AI·빅테크", "ADBE": "AI·빅테크", "PLTR": "AI·빅테크",
    # 금융
    "105560.KS": "금융", "055550.KS": "금융", "086790.KS": "금융",
    "316140.KS": "금융", "024110.KS": "금융", "323410.KS": "금융",
    "032830.KS": "금융",
    # 자동차
    "005380.KS": "자동차", "000270.KS": "자동차", "012330.KS": "자동차",
    "TSLA": "자동차",
    # 바이오
    "207940.KS": "바이오", "068270.KS": "바이오", "028300.KQ": "바이오",
    "196170.KQ": "바이오", "141080.KQ": "바이오",
    # 방산·에너지
    "012450.KS": "방산·에너지", "034020.KS": "방산·에너지",
    "015760.KS": "방산·에너지",
}

# Influential figures to watch for in news (optional, for event detection)
WATCHLIST_FIGURES: list[str] = [
    "제롬 파월", "파월", "Powell",
    "이창용", "한은 총재",
    "이복현", "금감원장",
    "젠슨 황", "Jensen Huang",
    "일론 머스크", "Elon Musk",
    "팀 쿡", "Tim Cook",
]

# yfinance ticker → Korean display name
TICKER_KR_NAME: dict[str, str] = {
    # KOSPI
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "373220.KS": "LG에너지솔루션",
    "207940.KS": "삼성바이오로직스",
    "005380.KS": "현대자동차",
    "000270.KS": "기아",
    "005490.KS": "POSCO홀딩스",
    "051910.KS": "LG화학",
    "068270.KS": "셀트리온",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "012330.KS": "현대모비스",
    "006400.KS": "삼성SDI",
    "066570.KS": "LG전자",
    "035720.KS": "카카오",
    "035420.KS": "NAVER",
    "028260.KS": "삼성물산",
    "086790.KS": "하나금융지주",
    "316140.KS": "우리금융지주",
    "034020.KS": "두산에너빌리티",
    "017670.KS": "SK텔레콤",
    "030200.KS": "KT",
    "034730.KS": "SK",
    "011170.KS": "롯데케미칼",
    "015760.KS": "한국전력",
    "032830.KS": "삼성생명",
    "012450.KS": "한화에어로스페이스",
    "010130.KS": "고려아연",
    "323410.KS": "카카오뱅크",
    "259960.KS": "크래프톤",
    # KOSDAQ
    "247540.KQ": "에코프로비엠",
    "086520.KQ": "에코프로",
    "028300.KQ": "HLB",
    "196170.KQ": "알테오젠",
    "141080.KQ": "리가켐바이오",
    "239890.KQ": "피엔에이치테크",
    "277810.KQ": "레인보우로보틱스",
    # HK / China 반도체
    "0981.HK": "SMIC (중신국제)",
    "1347.HK": "화홍반도체",
    "002371.SZ": "NAURA Technology",
    "600584.SS": "JCET Group",
    "603501.SS": "Will Semiconductor",
    "603986.SS": "GigaDevice",
}

# Korean company name → yfinance ticker (KOSPI/KOSDAQ)
KOSPI_TICKER_MAP: dict[str, str] = {
    # KOSPI 대형주
    "삼성전자": "005930.KS",
    "삼성": "005930.KS",
    "삼전": "005930.KS",  # 뉴스 헤드라인 약칭
    "SK하이닉스": "000660.KS",
    "하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "엘지에너지솔루션": "373220.KS",
    "엔솔": "373220.KS",  # 뉴스 헤드라인 약칭
    "삼성바이오로직스": "207940.KS",
    "삼성바이오": "207940.KS",
    "현대차": "005380.KS",
    "현대자동차": "005380.KS",
    "기아": "000270.KS",
    "기아차": "000270.KS",
    "POSCO홀딩스": "005490.KS",
    "포스코홀딩스": "005490.KS",
    "포스코": "005490.KS",
    "LG화학": "051910.KS",
    "엘지화학": "051910.KS",
    "셀트리온": "068270.KS",
    "KB금융": "105560.KS",
    "신한지주": "055550.KS",
    "신한금융": "055550.KS",
    "현대모비스": "012330.KS",
    "삼성SDI": "006400.KS",
    "삼성에스디아이": "006400.KS",
    "LG전자": "066570.KS",
    "엘지전자": "066570.KS",
    "카카오": "035720.KS",
    "네이버": "035420.KS",
    "NAVER": "035420.KS",
    "삼성물산": "028260.KS",
    "하나금융지주": "086790.KS",
    "하나금융": "086790.KS",
    "우리금융지주": "316140.KS",
    "우리금융": "316140.KS",
    "두산에너빌리티": "034020.KS",
    "SK텔레콤": "017670.KS",
    "SKT": "017670.KS",
    "KT": "030200.KS",
    "SK": "034730.KS",
    "롯데케미칼": "011170.KS",
    "한국전력": "015760.KS",
    "한전": "015760.KS",
    "삼성생명": "032830.KS",
    "한화에어로스페이스": "012450.KS",
    "한화에어로": "012450.KS",
    "고려아연": "010130.KS",
    "카카오뱅크": "323410.KS",
    "크래프톤": "259960.KS",
    # KOSDAQ
    "에코프로비엠": "247540.KQ",
    "에코프로": "086520.KQ",
    "HLB": "028300.KQ",
    "알테오젠": "196170.KQ",
    "리가켐바이오": "141080.KQ",
    "피엔에이치테크": "239890.KQ",
    "레인보우로보틱스": "277810.KQ",
}

# US company name → yfinance ticker (NASDAQ/NYSE)
NASDAQ_TICKER_MAP: dict[str, str] = {
    # Big Tech
    "애플": "AAPL",
    "Apple": "AAPL",
    "마이크로소프트": "MSFT",
    "Microsoft": "MSFT",
    "MS": "MSFT",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "Google": "GOOGL",
    "Alphabet": "GOOGL",
    "아마존": "AMZN",
    "Amazon": "AMZN",
    "엔비디아": "NVDA",
    "NVIDIA": "NVDA",
    "Nvidia": "NVDA",
    "메타": "META",
    "Meta": "META",
    "페이스북": "META",
    "테슬라": "TSLA",
    "Tesla": "TSLA",
    # Semiconductors
    "AMD": "AMD",
    "인텔": "INTC",
    "Intel": "INTC",
    "퀄컴": "QCOM",
    "Qualcomm": "QCOM",
    "브로드컴": "AVGO",
    "Broadcom": "AVGO",
    "TSMC": "TSM",
    "UMC": "UMC",
    "유나이티드마이크로일렉트로닉스": "UMC",
    "글로벌파운드리": "GFS",
    "글로벌파운드리스": "GFS",
    "GlobalFoundries": "GFS",
    "ARM": "ARM",
    # Cloud / SaaS
    "넷플릭스": "NFLX",
    "Netflix": "NFLX",
    "어도비": "ADBE",
    "Adobe": "ADBE",
    "세일즈포스": "CRM",
    "Salesforce": "CRM",
    "팔란티어": "PLTR",
    "Palantir": "PLTR",
    # Finance / Fintech
    "페이팔": "PYPL",
    "PayPal": "PYPL",
    "코인베이스": "COIN",
    "Coinbase": "COIN",
    # ETFs
    "QQQ": "QQQ",
    "SPY": "SPY",
    "SOXL": "SOXL",
    "TQQQ": "TQQQ",
}

# Hong Kong / China 반도체 종목명 → yfinance 티커 (SEHK: .HK, Shanghai: .SS, Shenzhen: .SZ)
HK_CN_TICKER_MAP: dict[str, str] = {
    "SMIC": "0981.HK",
    "중신국제": "0981.HK",
    "화홍반도체": "1347.HK",
    "Hua Hong": "1347.HK",
    "화홍": "1347.HK",
    "NAURA": "002371.SZ",
    "베이팡화촹": "002371.SZ",
    "JCET": "600584.SS",
    "창장과기": "600584.SS",
    "Will Semiconductor": "603501.SS",
    "웨이얼반도체": "603501.SS",
    "GigaDevice": "603986.SS",
    "자오이촹신": "603986.SS",
}

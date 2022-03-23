# 중도탈락 예측 모델 함수 관리

# 중도탈락 결과 칼럼은 1(중도탈락생), 0(재적생)으로 만든다.
# 데이터를 대학 뷰에서 추출 시 NULL이 존재하므로 필수 칼럼(학점, 시수, 평점 등은 NOT NULL)로 하고, 나머지 수치 칼럼은 0 처리함
query = """SELECT CASE WHEN GBN = 'OUT' THEN 1 ELSE 0 END AS GBN, YEAR, HAKBUN
     , HAKGI
     , SEX
     , DECODE(JOLUP_GUBUN,NULL,'기타',JOLUP_GUBUN) AS JOLUP_GUBUN
     , JUNHYUNG_HNAME
     , HAPKUK_GBN
     , HAKJUM, SISU, PYUNGJUM, KS, GK
     --, L_HAKJUM, L_SISU, L_PYUNGJUM, L_KS, L_GK
     , NVL(FOLL_CNT   ,0) FOLL_CNT
     , NVL(GWAMOK_CNT ,0) GWAMOK_CNT
     , NVL(F_CNT      ,0) F_CNT
     , NVL(GYNGO_CNT  ,0) GYNGO_CNT
     , NVL(HUHAK_CNT  ,0) HUHAK_CNT
     , NVL(JANGHAK_CNT,0) JANGHAK_CNT
  FROM KIRS1004
WHERE YEAR """
query_end = """
ORDER BY HAKBUN"""
    # 칼럼에 동일한 데이터만 존재하면 평균, 표준편차 오류 발생 : 해당 칼럼 제외

query_max_year = "SELECT MAX(YEAR) FROM KIRS1004"

excpet_list = ["GBN", "YEAR", "HAKBUN", "HAKGI", "SEX"]  # key_factor_extract 에서 활용


def getFactorList(v_cursor):
    # KIRS0007에서 PRED_FACTOR 가져오기 - db에 재적생,제적생이 모두 존재하는 테이블 및 관련 조인 쿼리에서 참조하는 모든 항목
    # 위 항목들은 공통코드 PRED_FACTOR에 사전에 등록한다.
    # 참조 파일에서 cursor가 선언되어야 함
    factorSql = """SELECT COM_COD, COM_NM, COM_UDC1, COM_UDC2, COM_UDC4 FROM KIRS0007 WHERE COML_COD = 'PRED_FACTOR' AND USE_YN = 'Y' ORDER BY SORT_ORDER"""
    v_cursor.execute(factorSql)
    factor_list = v_cursor.fetchall()

    v_modify_variables = []
    v_item_list = {}
    v_factorList = []

    for fc in factor_list:
        if fc[2] == "M":
            v_modify_variables.append(fc[0])
        elif fc[4] == "Y":
            v_factorList.append(fc[0])

        v_item_list[fc[0]] = fc[3]

    return [v_modify_variables, v_item_list, v_factorList]


def get_insert_text(len_col):
    ins_txt = ""
    for i in range(len_col):
        ins_txt += ":" + str(i+1) + ","
    return ins_txt


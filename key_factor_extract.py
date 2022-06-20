import cx_Oracle
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from predict_func import *
from sklearn.inspection import permutation_importance as pi


def keyFactorExtract(vVerGbn, vUserId):
    dsn = cx_Oracle.makedsn("ip", 1521, "sid")
    conn = cx_Oracle.connect("id", "pw", dsn)  # 해당 대학 db 계정으로 변경

    cursor = conn.cursor()
    modify_variables, column_decimal_list, factor_list = getFactorList(cursor)

    condYear = ""

    modelYearSql = """SELECT START_YEAR, END_YEAR FROM KIRS1000 WHERE PRED_TYPE = 'A'"""
    cursor.execute(modelYearSql)
    modelYearRange = cursor.fetchall()
    analCondYear = "BETWEEN " + modelYearRange[0][0] + " AND '" + modelYearRange[0][1] + "'"

    # query 조건에 각 칼럼별 not null 추가
    factor_cond_sql = ""
    for factor in factor_list:
        factor_cond_sql += " AND "+factor+" IS NOT NULL"

    df = pd.read_sql(query+analCondYear+factor_cond_sql+query_end, con=conn)

    df_dummy = pd.get_dummies(df, columns=modify_variables)
    # get_dummies : 범주형 데이터에 대해서 해당 칼럼의 각 범주를 1,0으로 수치화하여 칼럼을 생성함

    col_list = list(set(df_dummy.columns.tolist()) - set(excpet_list))  # 분석 칼럼만 추출
    features = df_dummy[col_list]

    # 중도탈락 결과 칼럼 할당
    out_student = df["GBN"]  # 결과 칼럼 데이터는 2개 이상이어야 함. 2개(이항), 3개이상(다항)

    train_features, test_features, train_labels, test_labels = train_test_split(features, out_student, random_state=42)

    #n_estimator = 35
    #n_depth = 9
    #n_split = 5

    from sklearn.ensemble import RandomForestClassifier
    # rf_model = RandomForestClassifier(n_estimators=n_estimator, max_depth=n_depth, min_samples_split=n_split, n_jobs=6, random_state=42)
    rf_model = RandomForestClassifier(n_jobs=6, random_state=42)
    rf_model.fit(train_features, train_labels)
    importances = rf_model.feature_importances_  # 주요 항목 추출

    # permutation importance 추출
    p_result = pi(rf_model, train_features, train_labels, scoring="f1", n_repeats=30, random_state=42, n_jobs=6)
    p_importances = p_result.importances_mean

    feat_imps = []
    for idx, feat in enumerate(col_list):
        try:
            decimalPlaces = column_decimal_list[feat]
        except KeyError:
            decimalPlaces = column_decimal_list[feat[:feat.rfind("_")]]
        feat_imps.append([feat, importances[idx], p_importances[idx], decimalPlaces])

    try:
        factorDelSql = """DELETE KIRS1002 WHERE VER_GBN = """+str(vVerGbn)
        cursor.execute(factorDelSql)
        # FACTOR 이름 및 확률 INSERT
        factorInsSql = """INSERT INTO KIRS1002 (VER_GBN, FACTOR_INPUT_CNAME, FACTOR_HNAME, FACTOR_MAP_COLUMN,""" + \
                       """FACTOR_RATIO_1, FACTOR_RATIO_2, DECIMAL_PLACES, KEY_FACTOR_YN,""" + \
                       """SORT_ORDER,INSERT_DAT, INSERT_EMP, UPDATE_DAT, UPDATE_EMP)""" + \
                       """ VALUES ("""+str(vVerGbn)+""", :1, NULL, NULL, :2, :3, :4, NULL, NULL""" + \
                       """, SYSDATE, '"""+vUserId+"""', SYSDATE, '"""+vUserId+"""')"""
        cursor.executemany(factorInsSql, feat_imps)
        # FACTOR_INPUT_CNAME을 한글명으로 업데이트
        tableUpdateSql = """UPDATE KIRS1002 SET FACTOR_HNAME = SF_KIRS1002(FACTOR_INPUT_CNAME, 'A')"""+\
                         """     , ORG_CNAME = SF_KIRS1002(FACTOR_INPUT_CNAME, 'B')"""+\
                         """     , DUMMY_DIV_DATA = SF_KIRS1002(FACTOR_INPUT_CNAME, 'C')"""+\
                         """WHERE VER_GBN = """+str(vVerGbn)
        cursor.execute(tableUpdateSql)
        cursor.close()
        conn.commit()
        print("factor_created")
    except cx_Oracle.DatabaseError as e:
        error, = e.args
        print(factorInsSql)
        print(error.code)
        print(error.message)
        print(error.context)
        cursor.close()
        conn.rollback()
    conn.close()


    # 요소 중요도 출력
    # ftImp_df = pd.DataFrame(feat_imps, columns=["Feature", "Importance"])
    # print(ftImp_df.sort_values("Importance", ascending=False))


def execPredFunc(argv):
    # argv[0] : 파일명
    if len(argv) == 3:
        # argv[1] : 버전구분, argv[2] : 유저id
        keyFactorExtract(argv[1], argv[2])
    elif len(argv) == 1:
        keyFactorExtract(101, "admin")  # 버전은 101부터


if __name__ == "__main__":
    execPredFunc(sys.argv)

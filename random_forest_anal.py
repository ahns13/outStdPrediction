import cx_Oracle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
from predict_func import *
from operator import itemgetter


def randomForestPred(vExecType, vCondYear, vVerGbn):
    # vExecType: 모델생성/예측실행 여부(A/B)

    dsn = cx_Oracle.makedsn("61.81.234.137", 1521, "COGDW")
    conn = cx_Oracle.connect("ansan", "ansan$#@!", dsn)
    cursor = conn.cursor()

    modify_variables, factor_list = itemgetter(0, 2)(getFactorList(cursor))

    condYear = " = '"+vCondYear+"'"

    # query 조건에 각 칼럼별 not null 추가
    factor_cond_sql = ""
    for factor in factor_list:
        factor_cond_sql += " AND " + factor + " IS NOT NULL"

    # 학기 제한 조건
    cur_year, cur_month = datetime.datetime.now().year, datetime.datetime.now().month
    input_hakgi = '2' if cur_month >= 12 else '1'

    cursor.execute(query_max_year)
    max_year = cursor.fetchall()[0][0]

    if cur_year == max_year:
        query_cond_hakgi = " AND HAKGI = '"+input_hakgi+"'"
    else:
        query_cond_hakgi = " AND HAKGI = '2'"

    global rf_model

    file_path = "D:\\python_outStdPrediction\\out_student_anal_v"+vVerGbn+".pkl"

    # key factor 가져오기
    keyFactorSql = """SELECT FACTOR_INPUT_CNAME, FACTOR_MAP_COLUMN FROM KIRS1002 """+\
                   """ WHERE VER_GBN = """+vVerGbn+""" AND KEY_FACTOR_YN = 'Y'"""+\
                   """ ORDER BY NVL(FACTOR_RATIO_2,0) DESC"""
    cursor.execute(keyFactorSql)
    keyFactorResult = cursor.fetchall()
    keyFactorItemList = []
    keyFactorColumnList = []

    for factor in keyFactorResult:
        keyFactorItemList.append(factor[0])  # 사용자 선택 주요 요인의 추출 칼럼
        keyFactorColumnList.append(factor[1])  # 주요 요인으로 분석한 결과 데이터가 들어갈 테이블의 매핑 칼럼

    if vExecType == "A":
        modelYearSql = """SELECT START_YEAR, END_YEAR FROM KIRS1000 WHERE PRED_TYPE = 'B'"""
        cursor.execute(modelYearSql)
        modelYearRange = cursor.fetchall()
        analCondYear = "BETWEEN "+modelYearRange[0][0]+" AND '"+modelYearRange[0][1]+"'"

        df = pd.read_sql(query+analCondYear+factor_cond_sql+query_cond_hakgi+query_end, con=conn)
        df_dummy = pd.get_dummies(df, columns=modify_variables)

        features = df_dummy[keyFactorItemList]
        out_student = df["GBN"]  # 결과 칼럼 데이터는 2개 이상이어야 함. 2개(이항), 3개이상(다항)
        # 분석할 데이터는 재적생과 제적생 모두 포함되는 전체 재학생 데이터이어야 함.

        train_features, test_features, train_labels, test_labels = train_test_split(features, out_student, random_state=42)

        # n_estimator = 35
        # n_depth = 9
        # n_split = 5

        from sklearn.ensemble import RandomForestClassifier
        # rf_model = RandomForestClassifier(n_estimators=n_estimator, max_depth=n_depth, min_samples_split=n_split, n_jobs=6, random_state=42)
        rf_model = RandomForestClassifier(n_jobs=6, random_state=42)
        # random_state 생략시 서버마다 생성된 결과가 다르게 나타나므로 주의
        rf_model.fit(train_features, train_labels)

        joblib.dump(rf_model, file_path)  # 파일 저장
        fileModDate = os.path.getmtime(file_path)
        rf_model_cdate = datetime.datetime.fromtimestamp(fileModDate).strftime("%Y-%m-%d %H:%M:%S")

        try:
            updateSql = """UPDATE KIRS0007 SET COM_UDC1 = '""" + rf_model_cdate + """' WHERE COML_COD = 'PRED_MODEL_CDATE' AND COM_COD = 'RF'"""
            # 모델 생성 일자를 공통코드에 업데이트한다.
            cursor.execute(updateSql)
            cursor.close()
            conn.commit()
            print("model_created")  # java에서 outReader.readLine()에 해당 문구가 전달됨.
        except cx_Oracle.DatabaseError as e:
            error, = e.args
            print(updateSql)
            print(error.code)
            print(error.message)
            print(error.context)
            cursor.close()
            conn.rollback()
        conn.close()

    else:
        if os.path.isfile(file_path):
            rf_model = joblib.load(file_path)
        else:
            print("no_model_file")

        df_pred = pd.read_sql(query + condYear + factor_cond_sql + query_cond_hakgi + query_end, con=conn)
        df_dummy = pd.get_dummies(df_pred, columns=modify_variables)

        pred_list = df_dummy[keyFactorItemList]
        # 아래 pred_list에 OUT_RESULT, OUT_RATIO 칼럼을 추가하는데 COPY를 하지 않고 추가하면 .loc[row_indexer,col_indexer] = value instead 오류 발생
        # 자신을 copy()로 할당하여 재 선언하면 오류 없음
        pred_list = pred_list.copy()

        rf_pred_result = rf_model.predict(pred_list)
        rf_pred_result_prb = rf_model.predict_proba(pred_list)
        pred_list["OUT_RESULT"] = rf_pred_result
        pred_list["OUT_RATIO"] = rf_pred_result_prb[:, 1]

        preColumnList = ["GBN", "YEAR", "HAKBUN", "HAKGI"]
        resultColumnList = ["OUT_RESULT", "OUT_RATIO"]
        restColumnList = ["VER_GBN", "INSERT_DAT", "INSERT_EMP", "UPDATE_DAT", "UPDATE_EMP"]
        insert_sql_columns = ",".join(preColumnList+keyFactorColumnList+resultColumnList+restColumnList)
        # concat : 두 dataframe을 합친다. axis=0이면 row로 추가, 1이면 column으로 추가
        insert_data = pd.concat([df_dummy[preColumnList], pred_list[keyFactorItemList + resultColumnList]], axis=1).values.tolist()
        var_columns_length = len(preColumnList+keyFactorColumnList+resultColumnList)  # 2: OUT_RESULT, OUT_RATIO
        # print(insert_data[:10])

        insertQuery = """INSERT INTO KIRS1001 ("""+insert_sql_columns+""") """+\
                      """VALUES ("""+get_insert_text(var_columns_length)+\
                      str(vVerGbn) + """, SYSDATE, 'admin', SYSDATE, 'admin')"""

        cursor.execute("DELETE FROM KIRS1001 WHERE YEAR" + condYear)

        try:
            cursor.executemany(insertQuery, insert_data)
            cursor.close()
            conn.commit()
            print("table_inserted")  # java에서 outReader.readLine()에 해당 문구가 전달됨.
        except cx_Oracle.DatabaseError as e:
            error, = e.args
            print(insertQuery)
            print(error.code)
            print(error.message)
            print(error.context)
            cursor.close()
            conn.rollback()
        conn.close()

        # plt.rcParams['font.family'] = 'NanumGothic'  # 한글 깨짐으로 인한 폰트 지정
        # # fig, ax = plt.subplots(ncols=1)
        # fig = plt.figure()
        # # confusion matrix : y축(y_true), x축(y_pred)
        # cm = confusion_matrix(np.array(pred_out_student), rf_pred_result)
        #
        # heatmap = sns.heatmap(cm, annot=True, annot_kws={"size": 13}, fmt="d",
        #                       xticklabels=["예측 In", "예측 Out"], yticklabels=["real In", "real Out"]).set(title="중도탈락 예측 현황")
        #
        # plt.show()


        # 예측한 데이터 추출
        # in_to_out, out_to_in = [], []  # 재학을 중도탈락으로, 중도탈락을 재학으로 : 실제 -> 예측
        # for idx, realVal in enumerate(pred_list["GBN"].values):
        #     if realVal == 1 and rf_pred_result[idx] == 0:  # out_to_in
        #         # print(idx, '번째 데이터 오류, 값(실값/예측): ', realVal, '/', rf_pred_result[idx], ' 예측확률: ', rf_pred_result_prb[idx])
        #         # print(pred_list.loc[idx])
        #         out_to_in.append(rf_pred_result_prb[idx])
        #     elif realVal == 0 and rf_pred_result[idx] == 1:  # in_to_out
        #         in_to_out.append(rf_pred_result_prb[idx])


def execPredFunc(argv):
    # argv[0] : 파일명
    if len(argv) > 2:
        # argv[1] : 모델생성/예측실행 여부(A/B), argv[2] : 년도, argv[3]: 버전
        randomForestPred(argv[1], argv[2], argv[3])
    elif len(argv) == 1:
        inputExecType = input("모델생성/예측실행[A/B] : ")
        inputYear = input("년도 : ")
        inputVerGbn = input("버전 : ")
        randomForestPred(inputExecType, inputYear, inputVerGbn)


if __name__ == "__main__":
    execPredFunc(sys.argv)

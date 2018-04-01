import os

def loadRisks(filename):
    lines = open(filename).readlines()
    while len(lines) > 0 and lines[0].lower().strip() == 'risk':
        lines = lines[1:]
    values = []
    for v in lines:
        try:
            values.append(int(v))
        except:
            print(v)
            return 'format error'
    return values

TRUTH_FILE = "./contest_private/test_risk.csv"
truth = loadRisks(TRUTH_FILE)

def evaluate(truth, pred):
    if len(truth) != len(pred):
        return -1, -1, -1

    mat = [ [0, 0], [0, 0] ]
    MRAE, cnt = 0, 0
    for (t, p) in zip(truth, pred):
        mat[int(t == 0)][int(p == 0)] += 1
        if t > 0:
            cnt += 1
            if p == -1:
                MRAE += 1
            else:
                MRAE += abs(p - t) / t

    MRAE /= cnt
    if mat[1][1] == 0:
        F1 = 0
    else:
        precision = float(mat[1][1]) / (mat[1][1] + mat[0][1])
        recall = float(mat[1][1]) / (mat[1][1] + mat[1][0])
        F1 = precision * recall * 2 / (precision + recall)

    return F1 + (1 - MRAE), F1, MRAE


SUBMISSION_FOLDER = "./submissions/"
results = []
for file in os.listdir(SUBMISSION_FOLDER):
    if file.endswith(".csv"):
        filename = os.path.join(SUBMISSION_FOLDER, file)
        submission_id = file.split('.')[0]
        pred = loadRisks(filename)

        if pred == 'format error':
            final_score = F1 = MRAE = -1
        else:
            final_score, F1, MRAE = evaluate(truth, pred)
        results.append((submission_id, final_score, F1, MRAE))

results = sorted(results, key = lambda x : -x[1])

out = open('standings.csv', 'w')
out.write('rank,submission_id,final_score,F1,MRAE\n')
print('rank,submission_id,final_score,F1,MRAE')
for i, (submission_id, final_score, F1, MRAE) in enumerate(results):
    values = [i + 1, submission_id, final_score, F1, MRAE]
    str_values = [str(value) for value in values]
    out.write(','.join(str_values) + '\n')
    print(','.join(str_values))
out.close()
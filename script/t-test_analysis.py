from scipy.stats import ttest_1samp
import numpy as np

# AIP 数据消融
raw_data = {
    'Origin Model': {
        'Accuracy': [0.7804],
        'F1 Score': [0.7143],
        'MCC': [0.5500]
    },
    'Model w/o SP': {
        'Accuracy': [0.7733, 0.7709, 0.7733, 0.7685, 0.7685],
        'F1 Score': [0.6940, 0.7130, 0.7108, 0.7230, 0.7034],
        'MCC': [0.5257, 0.5335, 0.5285, 0.5338, 0.5238]
    },
    'Model w/o NP': {
        'Accuracy': [0.7613365155131265, 0.7756563245823389, 0.7756563245823389, 0.7708830548926014, 0.7684964200477327],
        'F1 Score': [0.6913580246913581, 0.723529411764706, 0.7283236994219654, 0.7142857142857142, 0.7087087087087087],
        'MCC': [0.5089814009413177, 0.5385029482892649, 0.539077742651354, 0.528417993948469, 0.5234341310971747]
    },
    'Model w/o DDE': {
        'Accuracy': [0.7565632458233891, 0.7565632458233891, 0.7565632458233891, 0.7494033412887828, 0.7589498806682577],
        'F1 Score': [0.6946107784431139, 0.6946107784431139, 0.6946107784431139, 0.6728971962616823, 0.6966966966966968],
        'MCC': [0.4984196824182205, 0.4984196824182205, 0.4984196824182205, 0.48392094677657593, 0.5034068367012551]
    },
    'Model w/o CKSAAP': {
        'Accuracy': [0.7613365155131265, 0.7613365155131265, 0.7565632458233891, 0.7613365155131265, 0.7565632458233891],
        'F1 Score': [0.7191011235955056, 0.7206703910614525, 0.7102272727272727, 0.7175141242937854, 0.7102272727272727],
        'MCC': [0.5117754163705105, 0.5123727784685096, 0.5009220804273686, 0.5112253815238357, 0.5009220804273686]
    },
}
# AMP 数据消融
raw_data = {
    'Origin Model': {
        'Accuracy': [0.9173157579388855, 0.920311563810665, 0.920311563810665, 0.920910724985021, 0.920910724985021],
        'F1 Score': [0.9184397163120568, 0.9218106995884774, 0.9222676797194622, 0.9228971962616822, 0.9232558139534883],
        'MCC': [0.8349796810379649, 0.840713303371518, 0.8405386743710749, 0.8417269766134918, 0.8416842110991108]
    },
    'Model w/o SP': {
        'Accuracy': [0.9095266626722588, 0.9113241461953265, 0.9107249850209707, 0.9113241461953265, 0.9113241461953265],
        'F1 Score': [0.9123621590249564, 0.9143518518518519, 0.9137232194557037, 0.9142526071842411, 0.9142526071842411],
        'MCC': [0.8188957664350989, 0.8225208259328608, 0.8213133790492512, 0.8225067651569026, 0.8225067651569026]
    },
    'Model w/o NP': {
        'Accuracy': [0.9161174355901738, 0.9155182744158179, 0.914319952067106, 0.9131216297183943, 0.914319952067106],
        'F1 Score': [0.9184149184149184, 0.9182608695652174, 0.9172932330827067, 0.9160393746381007, 0.9172932330827067],
        'MCC': [0.8321017248267557, 0.8308994688345785, 0.8285293661107415, 0.8261126733188413, 0.8285293661107415]
    },
    'Model w/o DDE': {
        'Accuracy': [0.9155182744158179, 0.9149191132414619, 0.914319952067106, 0.9173157579388855, 0.9155182744158179],
        'F1 Score': [0.9147005444646098, 0.9141475211608222, 0.9138034960819772, 0.9172661870503598, 0.9153153153153154],
        'MCC': [0.8336341272964956, 0.8323534698779381, 0.8307692562717952, 0.8361501785622101, 0.8327469129474787]
    },
    'Model w/o CKSAAP': {
        'Accuracy': [0.9011384062312762, 0.9011384062312762, 0.8987417615338527, 0.8945476333133613, 0.896345116836429],
        'F1 Score': [0.9037900874635569, 0.9037900874635569, 0.9012273524254822, 0.8956109134045076, 0.897814530419374],
        'MCC': [0.8021271493946237, 0.8021271493946237, 0.7973713455209759, 0.7896517230921153, 0.7929879924674396]
    },
}

# AMP 结构消融
raw_data = {
    'Origin Model': {
        'Accuracy': [0.9173157579388855, 0.920311563810665, 0.920311563810665, 0.920910724985021, 0.920910724985021],
        'F1 Score': [0.9184397163120568, 0.9218106995884774, 0.9222676797194622, 0.9228971962616822, 0.9232558139534883],
        'MCC': [0.8349796810379649, 0.840713303371518, 0.8405386743710749, 0.8417269766134918, 0.8416842110991108]
    },
    'Model w/o RF': {
        'Accuracy': [0.9095266626722588, 0.9101258238466148, 0.9101258238466148, 0.9101258238466148, 0.9107249850209707],
        'F1 Score': [0.9099582587954681, 0.9108204518430439, 0.9110320284697508, 0.9111374407582937, 0.9112567004169149],
        'MCC': [0.8200430094752796, 0.8209965373679576, 0.8208214108433877, 0.8207409262299991, 0.8223392333559747]
    },
    'Model w/o XGBoost': {
        'Accuracy': [0.9035350509286998, 0.9029358897543439, 0.9035350509286998, 0.902336728579988, 0.902336728579988],
        'F1 Score': [0.903765690376569, 0.9041420118343196, 0.904109589041096, 0.9026865671641792, 0.9030339083878643],
        'MCC': [0.8082692999450807, 0.8062795203093877, 0.8079517772608001, 0.8057606802075526, 0.8054574189078413]
    },
    'Model w/o SVM': {
        'Accuracy': [0.9113241461953265, 0.9125224685440384, 0.9137207908927502, 0.9137207908927502, 0.9137207908927502],
        'F1 Score': [0.9131455399061033, 0.914918414918415, 0.9162790697674418, 0.9162790697674418, 0.9162790697674418],
        'MCC': [0.8226645663245462, 0.824906084462188, 0.8272908203479941, 0.8272908203479941, 0.8272908203479941]
    },
    'Model with LR': {
        'Accuracy': [0.9107249850209707, 0.908328340323547, 0.9101258238466148, 0.9095266626722588, 0.9095266626722588],
        'F1 Score': [0.9099697885196374, 0.9075528700906343, 0.9096385542168676, 0.9089813140446051, 0.9089813140446051],
        'MCC': [0.8238740946077999, 0.8190740545893669, 0.822298087563773, 0.8211711371343282, 0.8211711371343282]
    },
}


# raw_data = {
#     'Origin Model': {
#         'Accuracy': [0.7804],
#         'F1 Score': [0.7143],
#         'MCC': [0.5500]
#     },
#     'Model w/o RF': {
#         'Accuracy': [0.7565632458233891, 0.7565632458233891, 0.7494033412887828, 0.7804295942720764, 0.7494033412887828],
#         'F1 Score': [0.698224852071006, 0.6964285714285715, 0.6827794561933536, 0.6827794561933536, 0.6772151898734178],
#         'MCC': [0.49862799048089923, 0.4984981102370625, 0.48333720306132116, 0.48333720306132116, 0.5001848779776029]
#     },
#     'Model w/o XGBoost': {
#         'Accuracy': [0.7732696897374701, 0.7708830548926014, 0.7732696897374701, 0.7708830548926014, 0.7732696897374701],
#         'F1 Score': [0.7076923076923076, 0.7037037037037038, 0.7058823529411764, 0.7037037037037038, 0.7058823529411764],
#         'MCC': [0.5341988122192236, 0.5292844480656905, 0.534537123298962, 0.5292844480656905, 0.534537123298962]
#     },
#     'Model w/o SVM': {
#         'Accuracy': [0.7684964200477327, 0.7708830548926014, 0.7684964200477327, 0.7708830548926014, 0.7708830548926014],
#         'F1 Score': [0.7121661721068249, 0.7159763313609468, 0.7121661721068249, 0.7142857142857142, 0.7142857142857142],
#         'MCC': [0.5234577568158915, 0.5284715950916263, 0.5234577568158915, 0.528417993948469, 0.528417993948469]
#     },
#     'Model with LR': {
#         'Accuracy': [0.7517899761336515, 0.7517899761336515, 0.7517899761336515, 0.7517899761336515, 0.7517899761336515],
#         'F1 Score': [0.6556291390728478, 0.6556291390728478, 0.6556291390728478, 0.6556291390728478, 0.6556291390728478],
#         'MCC': [0.4944344449714865, 0.4944344449714865, 0.4944344449714865, 0.4944344449714865, 0.4944344449714865]
#     },
# }
# 初始化结果字典
results = {}

# 对每个模型进行一系列的t检验
for model_name, metrics in raw_data.items():
    if model_name == 'Origin Model':  # 跳过原始模型
        continue
    results[model_name] = {}
    for metric_name, values in metrics.items():
        # 对于每个指标，原模型的值作为总体均值
        population_mean = raw_data['Origin Model'][metric_name][0]
        # 执行单样本t检验
        t_stat, p_value = ttest_1samp(values, population_mean)
        # 如果p值小于0.05，则认为性能显著下降
        results[model_name][metric_name] = p_value < 0.05

print(results)
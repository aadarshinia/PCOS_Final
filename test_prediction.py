from predict import predict_pcos_risk



sample = {


"Age":25,

"BMI":28,

"Menstrual_Irregularity":1,

"Testosterone_Level(ng/dL)":55,

"Antral_Follicle_Count":15,

"Acne":1,

"Hair_Growth":1,

"Hair_Loss":0,

"Cycle_Length":45,

"LH":12,

"FSH":5


}



result = predict_pcos_risk(sample)


print(result)
import flask
import pickle
import pandas as pd
import numpy as np
import json

with open(f'model/score_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')





@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        data = pd.read_csv('model/merged_mldata2020v2.csv',delimiter = ',')
        outputs = []
        games = data.drop(columns =['vscore', 'hscore','winner','ou',"sWinner","fspread","fscore","vospread","vcspread"])
        games = pd.DataFrame(games)
        
        weekfilter = games['week'] == 1

        games = games[weekfilter]

        #data.corr()
        columnheads = ['temp','wind_mph','vdflg','hdflg','divgame','nsite','hospread','ouopen','hcspread','ouclose','vTOTAL.DVOA','vTOTAL.RNK','vOFF.RNK','vOFF.DVOA','vDEF.RNK','vDEF.DVOA','vST.RNK','vST.DVOA','hTOTAL.DVOA','hTOTAL.RNK','hOFF.RNK','hOFF.DVOA','hDEF.RNK','hDEF.DVOA','hST.RNK','hST.DVOA','vtsw','vtsl','vtst','htsw','htsl','htst','vtw','vtl','vtt','vts','htw','htl','htt','hts']

        games = np.hsplit(games,[3])
        tw = games[0]
        pdata = games[1]
        input_variables = pd.DataFrame(pdata, columns = columnheads,dtype=float)

        input_variables["prediction"] = model.predict(input_variables)

        output_info = input_variables[["hcspread","ouclose","prediction"]]
        output_info = tw.join(output_info)

        output_info["prediction"] = np.where(output_info["prediction"] < 0, output_info["vteam"] , output_info["hteam"])
        result = output_info.to_json(orient="table")
        parsed = json.loads(result)

    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('prediction.html',result=parsed["data"],)

if __name__ == '__main__':
    app.run()
import flask
import pickle
import pandas as pd

with open(f'model/score_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')





@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        predictions = []
        data = pd.read_csv('./data/merged_mldata2007-2019v2.csv',delimiter = ',')

        #data.corr()
        x = data.drop(columns =['vscore', 'hscore','winner','ou',"sWinner","fspread","fscore"])
        columnheads = ['temp','wind_mph','vdflg','hdflg','divgame','nsite','hospread','ouopen','hcspread','ouclose','vTOTAL.DVOA','vTOTAL.RNK','vOFF.RNK','vOFF.DVOA','vDEF.RNK','vDEF.DVOA','vST.RNK','vST.DVOA','hTOTAL.DVOA','hTOTAL.RNK','hOFF.RNK','hOFF.DVOA','hDEF.RNK','hDEF.DVOA','hST.RNK','hST.DVOA','vtsw','vtsl','vtst','htsw','htsl','htst','vtw','vtl','vtt','vts','htw','htl','htt','hts']


        x = pd.DataFrame(x)




        # Get the model's prediction
        prediction = model.predict(x)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
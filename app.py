import flask
import pickle
import pandas as pd

with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')





@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        temp = flask.request.form['temp']
        wind_mph = flask.request.form['wind_mph']
        vdflg = flask.request.form['vdflg']
        hdflg = flask.request.form['hdflg']
        divgame = flask.request.form['divgame']
        nsite = flask.request.form['nsite']
        hospread = flask.request.form['hospread']
        ouopen = flask.request.form['ouopen']
        hcspread = flask.request.form['hcspread']
        ouclose = flask.request.form['ouclose']
        vTOTALDVOA = flask.request.form['vTOTAL.DVOA']
        vTOTALRNK = flask.request.form['vTOTAL.RNK']
        vOFFRNK = flask.request.form['vOFF.RNK']
        vOFFDVOA = flask.request.form['vOFF.DVOA']
        vDEFRNK = flask.request.form['vDEF.RNK']
        vDEFDVOA = flask.request.form['vDEF.DVOA']
        vSTRNK = flask.request.form['vST.RNK']
        vSTDVOA = flask.request.form['vST.DVOA']
        hTOTALDVOA = flask.request.form['hTOTAL.DVOA']
        hTOTALRNK = flask.request.form['hTOTAL.RNK']
        hOFFRNK = flask.request.form['hOFF.RNK']
        hOFFDVOA = flask.request.form['hOFF.DVOA']
        hDEFRNK = flask.request.form['hDEF.RNK']
        hDEFDVOA = flask.request.form['hDEF.DVOA']
        hSTRNK = flask.request.form['hST.RNK']
        hSTDVOA = flask.request.form['hST.DVOA']
        vtsw = flask.request.form['vtsw']
        vtsl = flask.request.form['vtsl']
        vtst = flask.request.form['vtst']
        htsw = flask.request.form['htsw']
        htsl = flask.request.form['htsl']
        htst = flask.request.form['htst']
        vtw = flask.request.form['vtw']
        vtl = flask.request.form['vtl']
        vtt = flask.request.form['vtt']
        vts = flask.request.form['vts']
        htw = flask.request.form['htw']
        htl = flask.request.form['htl']
        htt = flask.request.form['htt']
        hts = flask.request.form['hts']



        # Make DataFrame for model
        inputVals = [[temp,wind_mph,vdflg,hdflg,divgame,nsite,hospread,ouopen,hcspread,ouclose,vTOTALDVOA,vTOTALRNK,vOFFRNK,vOFFDVOA,vDEFRNK,vDEFDVOA,vSTRNK,vSTDVOA,hTOTALDVOA,hTOTALRNK,hOFFRNK,hOFFDVOA,hDEFRNK,hDEFDVOA,hSTRNK,hSTDVOA,vtsw,vtsl,vtst,htsw,htsl,htst,vtw,vtl,vtt,vts,htw,htl,htt,hts]]

        columnheads = ['temp','wind_mph','vdflg','hdflg','divgame','nsite','hospread','ouopen','hcspread','ouclose','vTOTAL.DVOA','vTOTAL.RNK','vOFF.RNK','vOFF.DVOA','vDEF.RNK','vDEF.DVOA','vST.RNK','vST.DVOA','hTOTAL.DVOA','hTOTAL.RNK','hOFF.RNK','hOFF.DVOA','hDEF.RNK','hDEF.DVOA','hST.RNK','hST.DVOA','vtsw','vtsl','vtst','htsw','htsl','htst','vtw','vtl','vtt','vts','htw','htl','htt','hts']

        input_variables = pd.DataFrame(inputVals, columnheads,dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
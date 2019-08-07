from flask import Flask, Response
import json
import time
import stack_ensemble

app = Flask(__name__)


@app.route('/quant_forecast/eth_usdt', methods=['GET'])
def eth_price_forecast():
    current = time.time()
    model = stack_ensemble.STACK_ENSEMBLE()

    forecasts = model.main_execute()
    json_file = {
        'currencypair': 'eth_usdt',
        'forecasts': forecasts,
        'time_escaped': time.time()-current,
         }

    return Response(json.dumps(json_file), mimetype='application/json')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=12345)


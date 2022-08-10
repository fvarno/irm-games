First install the requirements:
`pip install fedsim>=0.3.0`

Then run the following command:

`fedsim-cli fed-learn --data-manager irmgame_defs:IRMDM --model cnn_mnist num_classes:2 num_channels:2 --epochs 2 --client-sample-rate 1 --n-clients 2 --log-freq 10`

Done!

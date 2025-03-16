python main.py --model predr \
               --epochs 100  \
               --chg_dims 8 4  \
               --dg_dims 8 4  \
               --activ relu  \
               --learning_rate 0.001  \
               --reduction_policy sum  \
               --log_path ./Log  \
               --ckpt_path ./Checkpoint  \
               --dataset_path ./Dataset  
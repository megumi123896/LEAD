python train_net.py \
    --config-file configs/referring_swin_tiny_eval.yaml \
    --num-gpus 8 --dist-url auto \
    --eval-only \
    MODEL.WEIGHTS "../checkpoints/upload_checkpoint/grefcoco_swin_tiny/LEAD_grefcoco_swin_tiny.pth" \
    OUTPUT_DIR "../results/LEAD_Grefcoco_Swin_tiny_eval"
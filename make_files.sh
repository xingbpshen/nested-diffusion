if [ "$DATASET" = "ChestXRay" ]; then
    export TMP_NAME=chest_x_ray
elif [ "$DATASET" = "ISICSkinCancer" ]; then
    export TMP_NAME=isic_skin_cancer
fi

if [ -d "diffusion/data/classification/pretrained/${TMP_NAME}_ckpt" ]; then
    rm -rf diffusion/data/classification/pretrained/${TMP_NAME}_ckpt/*
fi
mkdir -p diffusion/data/classification/pretrained/${TMP_NAME}_ckpt/

mv mapping/models/$DATASET/* diffusion/data/classification/pretrained/${TMP_NAME}_ckpt/
cp mapping/models/mlp.py diffusion/data/classification/pretrained/${TMP_NAME}_ckpt/

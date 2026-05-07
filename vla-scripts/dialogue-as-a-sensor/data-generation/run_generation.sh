export MUJOCO_GL=egl
# libEGL.so.1 / libGL.so.1 ship with the convai conda env (libegl/libgl
# from conda-forge); make them visible to PyOpenGL and mujoco.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/jc166/.conda/envs/convai}/lib:${LD_LIBRARY_PATH}"

python generate_vla_dataset.py \
    --output_dir ./my_data \
    --num_trials 10 \
    --num_videos 1 \
    --start_index 0
#data
python scripts/csv_to_npz.py \
--input_file /home/z/code/data/LAFAN1_Visualize/g1/dance1_subject2.csv \
--input_fps 30 --output_name dance1_subject2 --headless

python csv_to_npz.py --input_file /home/z/code/data/LAFAN1_Visualize/q2_19dof/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50

python scripts/csv_to_npz_q2.py \
--input_file /home/z/code/data/LAFAN1_Visualize/q2_19dof/dance1_subject2.csv \
--input_fps 30 --output_name dance1_subject2 --headless --frame_range 1 810


#play data
#g1
python scripts/replay_npz.py --registry_name=asdlkj/wandb-registry-motions/dance1_subject2

#q2

python scripts/replay_npz_q2.py --registry_name=asdlkj/wandb-registry-motions/dance1_subject2






/home/z/code/data/LAFAN1_Visualize/g1/dance1_subject2.csv 
dance1_subject1.csv         fallAndGetUp1_subject5.csv    jumps1_subject2.csv   walk1_subject5.csv
dance1_subject2.csv         fallAndGetUp2_subject2.csv    jumps1_subject5.csv   walk2_subject1.csv
dance1_subject3.csv         fallAndGetUp2_subject3.csv    run1_subject2.csv     walk2_subject3.csv
dance2_subject1.csv         fallAndGetUp3_subject1.csv    run1_subject5.csv     walk2_subject4.csv
dance2_subject2.csv         fight1_subject2.csv           run2_subject1.csv     walk3_subject1.csv
dance2_subject3.csv         fight1_subject3.csv           run2_subject4.csv     walk3_subject2.csv
dance2_subject4.csv         fight1_subject5.csv           sprint1_subject2.csv  walk3_subject3.csv
dance2_subject5.csv         fightAndSports1_subject1.csv  sprint1_subject4.csv  walk3_subject4.csv
fallAndGetUp1_subject1.csv  fightAndSports1_subject4.csv  walk1_subject1.csv    walk3_subject5.csv
fallAndGetUp1_subject4.csv  jumps1_subject1.csv           walk1_subject2.csv    walk4_subject1.csv




#train======================================================
# class Q2FlatWoStateEstimationEnvCfg(Q2FlatEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.observations.policy.motion_anchor_pos_b = None
#         self.observations.policy.base_lin_vel = None

# Tracking-Flat-G1-Wo-State-Estimation-v0  
# Tracking-Flat-G1-v0   
# Tracking-Flat-G1-Low-Freq-v0

python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name asdlkj/wandb-registry-motions/dance1_subject2 \
--headless --logger wandb --log_project_name g1 

# Tracking-Flat-Q2-Wo-State-Estimation-v0  
# Tracking-Flat-Q2-v0   
# Tracking-Flat-Q2-Low-Freq-v0



python scripts/rsl_rl/train.py --task=Tracking-Flat-Q2-Wo-State-Estimation-v0 \
--registry_name asdlkj/wandb-registry-motions/dance1_subject2 \
--headless --logger wandb --log_project_name q2 




#play
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
--wandb_path=aslk/g1/runs/04jjmfnr

python scripts/rsl_rl/play.py --task=Tracking-Flat-Q2-v0 --num_envs=2 \
--wandb_path=aslk/q2/runs/




###############################################################

python scripts/rsl_rl/play.py --task=Tracking-Flat-Q2-Wo-State-Estimation-v0 --num_envs=2 \
--wandb_path=aslk/q2/runs/u8o26zr7  #3000

6q78u2pz #810
u8o26zr7

# python scripts/rsl_rl/play.py --task=Tracking-Flat-Q2-Wo-State-Estimation-v0 --num_envs=2 \
# --motion_file assets/motions/motion.npz \
# --load_run 2025-10-16_13-49-59 \
# --checkpoint model_11500.pt \





wandb: ⭐️ View project at https://wandb.ai/aslk/g1
wandb: 🚀 View run at https://wandb.ai/aslk/g1/runs/833gtibo
wandb: 🚀 View run at https://wandb.ai/aslk/g1/runs/04jjmfnr



AttributeError: 'MotionOnPolicyRunner' object has no attribute 'obs_normalizer' #34
solved by this setting: isaac sim 4.5.0,isaac lab2.1.0,rslrl 2.3.3


git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
pip install -e .



# joint_stiffness: 157.914,157.914,157.914,157.914,157.914,78.957,78.957,157.914,157.914,16.581,16.581,157.914,157.914,16.581,16.581,33.162,33.162,16.581,16.581
# joint_damping: 10.053,10.053,10.053,10.053,10.053,5.027,5.027,10.053,10.053,1.056,1.056,10.053,10.053,1.056,1.056,2.111,2.111,1.056,1.056
# default_joint_pos: -0.332,-0.332,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,-0.000,0.524,0.524,0.000,0.000,-0.192,-0.192,0.000,0.000


00: 低速端等效惯量0.001kgm2   
01 02：低速端等效惯量4.2e-3 kg*m2     
03：低速端等效惯量0.02kgm2   
04：低速端等效惯量0.04 kg*m2  
05：低速端等效惯量0.0007kgm2   
06：低速端等效惯量0.012kgm2
python -B main_S3DIS_pretrain.py --gpu 0 --test_area 5 --log_dir '1pt-pre'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-pre'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-pre'

python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration1' --load_dir '1pt-pre' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-pre/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration1'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration1'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration2' --load_dir '1pt-iteration1' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration1/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration2'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration2'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration3' --load_dir '1pt-iteration2' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration2/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration3'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration3'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration4' --load_dir '1pt-iteration3' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration3/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration4'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration4'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration5' --load_dir '1pt-iteration4' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration4/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration5'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration5'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration6' --load_dir '1pt-iteration5' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration5/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration6'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration6'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration7' --load_dir '1pt-iteration6' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration6/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration7'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration7'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration8' --load_dir '1pt-iteration7' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration7/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration8'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration8'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration9' --load_dir '1pt-iteration8' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration8/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration9'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration9'
python -B main_S3DIS_Mstep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration10' --load_dir '1pt-iteration9' --pseudo_label_path './experiment/S3DIS/1_points_/1pt-iteration9/prediction/pseudo_label'
python -B main_S3DIS_Estep.py --gpu 0 --test_area 5 --log_dir '1pt-iteration10'
python -B main_S3DIS_test.py --gpu 0 --test_area 5 --log_dir '1pt-iteration10'
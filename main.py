import load
import data
from tcn import *
from transformer import *  
import torch
import torchvision.transforms as transforms
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse  
from tqdm import tqdm  # tqdm 추가

def set_seed(seed: int):
    """
    랜덤 시드 고정 함수.

    Parameters
    ----------
    seed : int
        고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, val_loader, criterion, device='cpu', model_type='tcn', epoch=0):
    model.eval()
    total_loss = 0.0

    # tqdm으로 진행 상황 표시
    val_loop = tqdm(val_loader, desc=f"Validation", leave=False)

    with torch.no_grad():
        for inputs, labels in val_loop:
            # Move inputs and labels to the selected device
            inputs, labels = inputs.to(device), labels.to(device)

            # 모델 타입에 따라 전처리 및 추론 방식 적용
            if model_type == 'tcn':
                # TCN용 처리 - Add channel dimension to inputs
                inputs = inputs.transpose(1, 2)  # (batch_size, 5, 30)
                outputs = model(inputs)
                outputs = outputs[:, :, -1]
            elif model_type == 'transformer':
                # Transformer용 처리
                batch_size = inputs.size(0)
                seq_len = inputs.size(1)  # 시퀀스 길이
                feature_size = inputs.size(2)  # 특성 수
                
                # 배치 내 모든 샘플을 처리하는 리스트
                outputs_list = []
                
                for i in range(batch_size):
                    # 각 샘플 처리
                    x = inputs[i]  # (seq_len, features)
                    
                    # 원본 코드와 일치하도록 3개의 연속된 시간 단계 사용
                    # 시퀀스 길이가 충분한 경우에만 처리
                    if seq_len >= 3:
                        # 각 시간 단계에 대해 처리 (원본 코드는 각 시간 단계마다 처리)
                        # 여기서는 평가를 위해 마지막 3개 시간 단계만 사용
                        last_steps = x[-3:, :]  # (3, features)
                        
                        # 4차원으로 변환 (1, 1, 3, features)
                        x_reshaped = last_steps.unsqueeze(0).unsqueeze(0)
                        
                        # 시간 단계 t는 시퀀스 내 위치 (여기서는 마지막 위치)
                        t = seq_len - 2  # 중간 위치 (t-1, t, t+1에서 t)
                        
                        # 모델 적용
                        output = model(x_reshaped, t)
                    else:
                        # 시퀀스 길이가 3보다 작은 경우 패딩 처리
                        padded_x = torch.zeros(3, feature_size, device=device)
                        padded_x[-seq_len:, :] = x
                        
                        # 4차원으로 변환
                        x_reshaped = padded_x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, features)
                        
                        # 시간 단계 t는 시퀀스 내 위치
                        t = max(1, seq_len - 1)  # 최소 1 (중간 위치)
                        
                        # 모델 적용
                        output = model(x_reshaped, t)
                    
                    outputs_list.append(output)
                
                # 모든 샘플의 결과를 하나의 텐서로 결합
                outputs = torch.cat(outputs_list, dim=0)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # tqdm 진행 상황 업데이트
            val_loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Define a function to train the model and save the best one
def train_and_save_best_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu', save_path="best_model.pth", model_type='tcn', patience=10, min_delta=0.0001, scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    
    # Early stopping 관련 변수
    counter = 0
    early_stop = False
    
    # tqdm으로 에포크 진행 상황 표시
    epoch_loop = tqdm(range(epochs), desc="Training")
    
    for epoch in epoch_loop:
        # Training phase
        model.train()
        total_loss = 0.0
        
        # 원본 코드와 유사하게 각 배치를 독립적으로 처리
        # tqdm으로 배치 진행 상황 표시
        batch_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (inputs, labels) in batch_loop:
            # Move inputs and labels to the selected device
            inputs, labels = inputs.to(device), labels.to(device)

            # 모델 타입에 따라 전처리 및 추론 방식 적용
            if model_type == 'tcn':
                # TCN용 처리 - Add channel dimension to inputs
                inputs = inputs.transpose(1, 2)  # (batch_size, 5, 30)
                outputs = model(inputs)
                outputs = outputs[:, :, -1]
                
                # 손실 계산 및 역전파
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # tqdm 진행 상황 업데이트
                batch_loop.set_postfix(loss=loss.item())
                
            elif model_type == 'transformer':
                # Transformer용 처리
                batch_size = inputs.size(0)
                seq_len = inputs.size(1)  # 시퀀스 길이
                feature_size = inputs.size(2)  # 특성 수
                
                # 원본 코드와 유사하게 각 샘플을 독립적으로 처리
                batch_loss = 0.0
                
                for i in range(batch_size):
                    # 각 샘플 처리
                    x = inputs[i]  # (seq_len, features)
                    y = labels[i]  # 단일 RUL 값
                    
                    # 원본 코드와 일치하도록 각 시간 단계에 대해 처리
                    # 여기서는 시퀀스 길이가 충분한 경우에만 처리
                    if seq_len >= 3:
                        # 각 시간 단계에 대해 처리 (원본 코드는 각 시간 단계마다 처리)
                        # 여기서는 시간 효율성을 위해 마지막 3개 시간 단계만 사용
                        sample_loss = 0.0
                        
                        # 마지막 3개 시간 단계에 대해 처리
                        for t in range(max(0, seq_len - 3), seq_len):
                            # 현재 시간 단계 t를 중심으로 3개의 연속된 시간 단계 선택
                            t_start = max(0, t - 1)
                            t_end = min(seq_len, t + 2)
                            
                            # 3개의 시간 단계가 되도록 패딩
                            if t_end - t_start < 3:
                                if t_start == 0:  # 시작 부분인 경우
                                    t_end = min(seq_len, 3)
                                else:  # 끝 부분인 경우
                                    t_start = max(0, seq_len - 3)
                            
                            # 3개의 연속된 시간 단계 선택
                            steps = x[t_start:t_end, :]
                            
                            # 필요한 경우 패딩
                            if steps.size(0) < 3:
                                padded_steps = torch.zeros(3, feature_size, device=device)
                                padded_steps[-steps.size(0):, :] = steps
                                steps = padded_steps
                            
                            # 4차원으로 변환 (1, 1, 3, features)
                            x_reshaped = steps.unsqueeze(0).unsqueeze(0)
                            
                            # 모델 적용 - t는 시퀀스 내 위치
                            output = model(x_reshaped, t)
                            
                            # 손실 계산 (각 시간 단계마다)
                            step_loss = criterion(output, y.unsqueeze(0))
                            sample_loss += step_loss.item()
                            
                            # 역전파 (원본 코드는 각 시간 단계마다 손실 누적)
                            step_loss.backward()
                        
                        # 샘플의 평균 손실
                        sample_loss /= min(3, seq_len)
                        batch_loss += sample_loss
                    else:
                        # 시퀀스 길이가 3보다 작은 경우 패딩 처리
                        padded_x = torch.zeros(3, feature_size, device=device)
                        padded_x[-seq_len:, :] = x
                        
                        # 4차원으로 변환
                        x_reshaped = padded_x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, features)
                        
                        # 시간 단계 t는 시퀀스 내 위치
                        t = max(1, seq_len - 1)  # 최소 1 (중간 위치)
                        
                        # 모델 적용
                        output = model(x_reshaped, t)
                        
                        # 손실 계산
                        sample_loss = criterion(output, y.unsqueeze(0))
                        batch_loss += sample_loss.item()
                        
                        # 역전파
                        sample_loss.backward()
                
                # 배치의 평균 손실
                batch_loss /= batch_size
                total_loss += batch_loss
                
                # tqdm 진행 상황 업데이트
                batch_loop.set_postfix(loss=batch_loss)
                
                # 원본 코드와 유사하게 배치 처리 후 옵티마이저 스텝
                optimizer.step()
                optimizer.zero_grad()
            
        # 에포크의 평균 손실
        avg_train_loss = total_loss / len(train_loader)
        
        # 학습률 스케줄러 적용 (있는 경우)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # tqdm 설명 업데이트
            epoch_loop.set_description(f"Training (lr={current_lr:.6f})")

        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion, device, model_type, epoch)
        
        # tqdm 진행 상황 업데이트
        epoch_loop.set_postfix(train_loss=avg_train_loss, val_loss=val_loss)
        
        # 콘솔에 자세한 정보 출력
        print(f"\nEpoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping 체크
        if val_loss < best_val_loss - min_delta:
            # 성능이 개선되었을 때
            best_val_loss = val_loss
            counter = 0
            # 최고 성능 모델 저장
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with loss: {best_val_loss:.4f}")
        else:
            # 성능이 개선되지 않았을 때
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                early_stop = True
                break
    
    if not early_stop:
        print(f"Training complete after full {epochs} epochs. Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"Training stopped early after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")

def predict_and_evaluate(model, x_test, y_test, device, model_type, now):
    model.eval()
    predictions = []
    
    # tqdm으로 테스트 진행 상황 표시
    test_loop = tqdm(range(len(x_test)), desc="Testing")
    
    with torch.no_grad():
        for i in test_loop:
            # x_test[i] -> shape: (seq_len, num_features)
            if model_type == 'tcn':
                x = torch.tensor(x_test[i], dtype=torch.float, device=device).unsqueeze(0)  # (1, seq_len, num_features)
                x = x.transpose(1, 2)  # (1, num_features, seq_len)
                output = model(x)      # (1, 1, seq_len)
                # 마지막 시점의 예측만 사용
                output = output[:, :, -1]  # (1, 1)
                predictions.append(output.item())
            elif model_type == 'transformer':
                x = torch.tensor(x_test[i], dtype=torch.float, device=device)  # (seq_len, num_features)
                seq_len = x.size(0)
                feature_size = x.size(1)
                
                # 원본 코드와 일치하도록 3개의 연속된 시간 단계 사용
                if seq_len >= 3:
                    # 마지막 3개 시간 단계 선택
                    last_steps = x[-3:, :]  # (3, features)
                    
                    # 4차원으로 변환 (1, 1, 3, features)
                    x_reshaped = last_steps.unsqueeze(0).unsqueeze(0)
                    
                    # 시간 단계 t는 시퀀스 내 위치 (여기서는 마지막 위치)
                    t = seq_len - 2  # 중간 위치 (t-1, t, t+1에서 t)
                    
                    # 모델 적용
                    output = model(x_reshaped, t)
                else:
                    # 시퀀스 길이가 3보다 작은 경우 패딩 처리
                    padded_x = torch.zeros(3, feature_size, device=device)
                    padded_x[-seq_len:, :] = x
                    
                    # 4차원으로 변환
                    x_reshaped = padded_x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, features)
                    
                    # 시간 단계 t는 시퀀스 내 위치
                    t = max(1, seq_len - 1)  # 최소 1 (중간 위치)
                    
                    # 모델 적용
                    output = model(x_reshaped, t)
                
                predictions.append(output.item())
            
            # tqdm 진행 상황 업데이트
            if i % 10 == 0:  # 10개마다 업데이트
                test_loop.set_postfix(sample=f"{i}/{len(x_test)}")
    
    # pandas.Series인 y_test와 예측값 리스트 predictions를 합쳐 DataFrame 생성
    df_pred = pd.DataFrame({
        'Actual_RUL': y_test.values, 
        'Predicted_RUL': predictions
    })

    # Actual_RUL 기준으로 내림차순 정렬
    df_pred = df_pred.sort_values(by='Actual_RUL', ascending=False).reset_index(drop=True)

    # test_loss (MSE) 계산
    test_loss = np.mean((df_pred['Actual_RUL'] - df_pred['Predicted_RUL']) ** 2)
    print("Test MSE:", test_loss)

    # 필요 시 RMSE로도 계산 가능
    test_rmse = np.sqrt(test_loss)
    print("Test RMSE:", test_rmse)

    # CSV로 저장
    df_pred.to_csv(f'predictions_{model_type}_{now}.csv', index=False)
    print(f"Predictions saved to predictions_{model_type}_{now}.csv.")

    # 그래프 출력
    plt.figure(figsize=(10,6))
    plt.plot(df_pred['Actual_RUL'], label='Actual RUL', marker='o')
    plt.plot(df_pred['Predicted_RUL'], label='Predicted RUL', marker='x')
    # 제목에 RMSE 값 추가
    plt.title(f'RUL Prediction - {model_type.upper()} (Test RMSE: {test_rmse:.4f})')
    plt.xlabel('Index')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    
    # 그래프에 RMSE 값 텍스트로 추가 (오른쪽 상단)
    plt.text(0.95, 0.95, f'RMSE: {test_rmse:.4f}', 
             transform=plt.gca().transAxes, 
             horizontalalignment='right', 
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(f'RUL_Prediction_{model_type}_{now}.png')
    plt.show()
    
    return test_rmse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RUL Prediction with TCN or Transformer')
    parser.add_argument('--model', type=str, default='tcn', choices=['tcn', 'transformer'],
                      help='Model type: tcn or transformer (default: tcn)')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset name (default: FD001)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    # Transformer 모델 관련 추가 매개변수 - 원본 코드와 일치하도록 기본값 수정
    parser.add_argument('--d_model', type=int, default=128, help='Transformer 임베딩 차원 (default: 128)')
    parser.add_argument('--heads', type=int, default=4, help='Transformer 어텐션 헤드 수 (default: 4)')
    parser.add_argument('--n_layers', type=int, default=2, help='Transformer 인코더 레이어 수 (default: 2)')
    parser.add_argument('--seq_len', type=int, default=30, help='시퀀스 길이 (default: 30)')
    # Early stopping 관련 매개변수
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change in validation loss to be considered as improvement (default: 0.0001)')
    # 학습률 스케줄러 관련 매개변수
    parser.add_argument('--lr_scheduler', action='store_true', help='학습률 스케줄러 사용 여부')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup 스텝 수 (default: 4000)')
    
    args = parser.parse_args()
    
    # 현재 날짜와 시간을 now 변수로 생성
    now = datetime.datetime.now().strftime("%m%d%H%M")

    # Set random seed for reproducibility
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Data loading parameters
    dataset = args.dataset
    # sensors to work with: T30, T50, P30, PS30, phi
    sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
    # windows length - transformer에는 더 긴 시퀀스가 도움이 됨
    sequence_length = args.seq_len if args.model == 'transformer' else 15
    # smoothing intensity
    alpha = 0.1
    # max RUL
    threshold = 125

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load.get_data(dataset, sensors, 
    sequence_length, alpha, threshold)

    # Create data loaders
    batch_size = args.batch_size
    n_epochs = args.epochs
    n_workers = 8  # 병렬 처리 워커 수 증가
    tf = transforms.Compose([data.NpToTensor()])
    train_loader = data.get_dataloader(x_train, y_train, tf, None, True, batch_size, n_workers)
    val_loader = data.get_dataloader(x_val, y_val, tf, None, True, batch_size, n_workers)

    # Model initialization based on user selection
    dropout = args.dropout
    m = len(sensors)  # number of features
    
    if args.model == 'tcn':
        # TCN 모델 정의 (출력 채널을 1로 설정해 최종 시계열의 마지막 시점이 RUL을 예측)
        model = TemporalConvNet(
            num_inputs=m,      # 입력 특성 수 (5)
            num_channels=[32, 16, 1],  # 중간 채널 크기 조절 가능
            kernel_size=3,
            dropout=dropout
        ).to(device)
        
        # Initialize weights
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        model_type = 'tcn'
        
    elif args.model == 'transformer':
        # Transformer 모델 정의 - 원본 코드와 일치하도록 수정
        d_model = args.d_model  # dimension in encoder (128)
        heads = args.heads      # number of heads in multi-head attention (4)
        N = args.n_layers       # number of encoder layers (2)
        
        print(f"Transformer 설정: d_model={d_model}, heads={heads}, layers={N}, features={m}, seq_len={sequence_length}")
        
        # 원본 코드와 일치하는 Transformer 모델 초기화
        model = Transformer(
            m=m,
            d_model=d_model,
            N=N,
            heads=heads,
            dropout=dropout
        ).to(device)
        
        # Initialize weights (원본 코드와 동일하게 Xavier 초기화 적용)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        model_type = 'transformer'
    
    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    
    # 원본 코드와 일치하도록 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습률 스케줄러 설정 (선택적)
    scheduler = None
    if args.lr_scheduler:
        # 학습 중 학습률 감쇠를 위한 스케줄러
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    # Train and save the best model
    print(f"Starting {model_type.upper()} training...")
    train_and_save_best_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        epochs=n_epochs, 
        device=device, 
        save_path=f"best_{model_type}_model_{now}.pth",
        model_type=model_type,
        patience=args.patience,
        min_delta=args.min_delta,
        scheduler=scheduler
    )

    # 모델 불러오기
    model.load_state_dict(torch.load(f"best_{model_type}_model_{now}.pth"))
    model.to(device)
    
    # 예측 및 평가
    test_rmse = predict_and_evaluate(model, x_test, y_test, device, model_type, now)
    print(f"Final {model_type.upper()} Test RMSE: {test_rmse}")

if __name__ == "__main__":
    main()
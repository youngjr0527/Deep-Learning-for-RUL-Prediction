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

    with torch.no_grad():
        for inputs, labels in val_loader:
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
                    
                    # 모델 적용 - 시간 정보로 epoch 전달
                    output = model(x.unsqueeze(0), epoch)  # 배치 차원 추가 (1, seq_len, features)
                    outputs_list.append(output)
                
                # 모든 샘플의 결과를 하나의 텐서로 결합
                outputs = torch.cat(outputs_list, dim=0)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Define a function to train the model and save the best one
def train_and_save_best_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu', save_path="best_model.pth", model_type='tcn', patience=10, min_delta=0.0001, scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    
    # Early stopping 관련 변수
    counter = 0
    early_stop = False
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
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
                # 배치 내 모든 샘플을 처리하는 리스트
                outputs_list = []
                
                for i in range(batch_size):
                    # 각 샘플을 평탄화하고 배치 차원 추가
                    x = inputs[i]  # (seq_len, features)
                    
                    # 디버깅 정보 (필요시 주석 해제)
                    # print(f"샘플 {i} 입력 형태: {x.shape}")
                    
                    # 모델 적용 - 시간 정보를 인덱스로 활용
                    output = model(x.unsqueeze(0), epoch)  # 배치 차원 추가 (1, seq_len, features)
                    outputs_list.append(output)
                
                # 모든 샘플의 결과를 하나의 텐서로 결합
                outputs = torch.cat(outputs_list, dim=0)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for Transformer
            if model_type == 'transformer':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
            optimizer.step()

            total_loss += loss.item()
            
        # 학습률 스케줄러 적용 (있는 경우)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.8f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion, device, model_type, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}")

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
    
    with torch.no_grad():
        for i in range(len(x_test)):
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
                # 배치 차원 추가
                x = x.unsqueeze(0)  # (1, seq_len, num_features)
                # 예측을 위해 t=0 사용
                output = model(x, 0)
                predictions.append(output.item())
    
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
    # Transformer 모델 관련 추가 매개변수 
    parser.add_argument('--d_model', type=int, default=256, help='Transformer 임베딩 차원 (default: 256)')
    parser.add_argument('--heads', type=int, default=8, help='Transformer 어텐션 헤드 수 (default: 8)')
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer 인코더 레이어 수 (default: 4)')
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
    n_workers = 4  # 병렬 처리 워커 수 증가
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
        # Transformer 모델 정의 - 매개변수를 커맨드 라인 인자로부터 가져옴
        d_model = args.d_model  # dimension in encoder (256 권장)
        heads = args.heads      # number of heads in multi-head attention (8 권장)
        N = args.n_layers       # number of encoder layers (4 권장)
        
        print(f"Transformer 설정: d_model={d_model}, heads={heads}, layers={N}, features={m}, seq_len={sequence_length}")
        
        model = Transformer(
            m=m,
            d_model=d_model,
            N=N,
            heads=heads,
            dropout=dropout
        ).to(device)
        
        # Initialize weights (Transformer도 Xavier 초기화 적용)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        model_type = 'transformer'
    
    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    
    # Transformer에는 낮은 학습률과 가중치 감쇠 적용
    if args.model == 'transformer':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습률 스케줄러 설정 (Transformer에만 적용)
    scheduler = None
    if args.model == 'transformer' and args.lr_scheduler:
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
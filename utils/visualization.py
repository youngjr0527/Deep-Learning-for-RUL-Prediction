import matplotlib.pyplot as plt


def visualize(result, rmse, model_name='Transformer'):
    """
    시각화 함수: 실제 RUL과 예측 RUL을 시각화합니다.
    
    Args:
        result: 결과 데이터프레임 (실제 RUL과 예측 RUL 포함)
        rmse: Root Mean Squared Error 값
        model_name: 모델 이름 (기본값: 'Transformer')
    """
    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.axvline(x=100, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data')
    plt.plot(pred_rul, label='Predicted Data')
    plt.title(f'RUL Prediction on CMAPSS Data using {model_name}')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig(f'RUL_Prediction_{model_name}.png')
    plt.show() 
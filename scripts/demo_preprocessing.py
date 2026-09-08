import torch
from defense_repro.preprocessing import (
    grayscale, smooth_image, fft_features,
    kmeans_binary_relevance,
)
from defense_repro.platform import single_pass_cluster, tfidf


def main():
    torch.manual_seed(0)

    img = torch.rand(4, 3, 32, 32)
    gray = grayscale(img)
    smooth = smooth_image(gray)

    wav = torch.randn(4, 1024)
    spec = fft_features(wav)

    features = torch.cat([
        torch.randn(32, 4) - 2,
        torch.randn(32, 4) + 2,
    ])
    labels, centers = kmeans_binary_relevance(features)

    news = torch.randn(20, 16)
    sp_labels, _ = single_pass_cluster(news, cosine_threshold=0.6)

    docs = [
        ["食品", "安全", "风险", "食品"],
        ["企业", "食品", "召回"],
        ["娱乐", "电影", "明星"],
    ]
    scores = tfidf(docs)

    print("gray:", tuple(gray.shape), "smooth:", tuple(smooth.shape))
    print("fft:", tuple(spec.shape))
    print("relevance clusters:", torch.bincount(labels).tolist())
    print("single-pass clusters:", int(sp_labels.max()) + 1)
    print("tfidf doc0:", scores[0])


if __name__ == "__main__":
    main()

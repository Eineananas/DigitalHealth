file_path <- "E:/SHARE/A159/ls_free_text_result_0.csv.gz"
t02 <- gzfile(file_path, "rt")
T02 <- read.table(t02, 
                  header = TRUE, 
                  sep = "\01", 
                  quote = "", 
                  comment.char = "#", 
                  fill = TRUE)

# 2. 关闭文件连接
close(t02)

# 3. 输出总数据量
cat("📊 总数据量:", nrow(T02), "行\n")
cat("📋 列数:", ncol(T02), "列\n")
cat("前几行预览:\n")
print(head(T02, 3))

# 4. 随机抽取10000条数据
if(nrow(T02) >= 10000) {
  set.seed(42)  # 固定随机种子，保证可重复
  sample_indices <- sample(1:nrow(T02), 10000)
  T02_sample <- T02[sample_indices, ]
  cat("\n🎲 已随机抽取 10,000 条数据\n")
} else {
  T02_sample <- T02
  cat("\n⚠️  数据不足10,000条，使用全部", nrow(T02), "条\n")
}

# 5. 保存到普通CSV
output_path <- "sampled_10000.csv"
write.csv(T02_sample, output_path, row.names = FALSE, quote = FALSE)

cat("✅ 已保存到:", output_path, "\n")
cat("🎉 处理完成！\n")

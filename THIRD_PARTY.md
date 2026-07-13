# Third-party dependencies

| Dependency | Repository | Current pinned commit |
|---|---|---|
| GoogleTest | https://github.com/google/googletest.git | `f132c893119698e10daef8525d0ad7a3f05176f2` |
| Google Benchmark | https://github.com/google/benchmark.git | `c4114ca2b76eefdb48222abff96c12160614b737` |
| spdlog | https://github.com/gabime/spdlog.git | `2ee3cf8204ed5048627644e00a51a7d93fbc4786` |

仓库根目录的 `.gitmodules` 记录 URL，Gitlink 记录具体版本。初始化或更新依赖：

```bash
git submodule update --init --recursive
git submodule update --remote --merge  # 仅在需要升级版本时执行
```

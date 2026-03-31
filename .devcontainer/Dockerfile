FROM node:20

ARG TZ
ENV TZ="$TZ"

# Install system deps + iptables/ipset for firewall
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  rsync \
  build-essential \
  libffi-dev \
  libssl-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv, then use uv to install Python 3.12
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && cp /root/.local/bin/uv /usr/local/bin/uv \
  && cp /root/.local/bin/uvx /usr/local/bin/uvx \
  && uv python install 3.12 \
  && ln -sf $(uv python find 3.12) /usr/local/bin/python3 \
  && ln -sf $(uv python find 3.12) /usr/local/bin/python

# Reuse the existing node user (UID/GID 1000), rename to dev
ARG USERNAME=node
RUN usermod -s /bin/zsh $USERNAME \
  && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Ensure npm global dir
RUN mkdir -p /usr/local/share/npm-global && chown -R $USERNAME:$USERNAME /usr/local/share/npm-global

# Persist shell history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true

# Create workspace and config dirs
RUN mkdir -p /workspace /home/node/.claude \
  && chown -R $USERNAME:$USERNAME /workspace /home/node/.claude

WORKDIR /workspace

# Install git-delta
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

USER root
RUN ln -s /home/node /home/fflores
USER $USERNAME

# NPM global path
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin:/home/$USERNAME/.local/bin

ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Install zsh with plugins
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude Code (native installer) and tsx (plugin dependency)
RUN npm uninstall -g @anthropic-ai/claude-code 2>/dev/null || true \
  && curl -fsSL https://claude.ai/install.sh | bash \
  && npm install -g tsx

# Copy and set up firewall script
COPY init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh \
  && echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/$USERNAME-firewall \
  && chmod 0440 /etc/sudoers.d/$USERNAME-firewall
USER $USERNAME

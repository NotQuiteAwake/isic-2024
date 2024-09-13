#!/bin/bash

# setup script for cloud GPU instance (experimental, partially untested)
# should work on Ubuntu 22.04 LTS from the PyTorch docker image at least
# must be run as root

set -e

USR="${1:-kaggle}"
USR_HOME="/home/$USR/"
WORK_DIR="$USR_HOME/workspace/"
GIT_DIR="$WORK_DIR/kaggle_2024/"

echo "set locale"
sed -i 's/# en_GB.UTF-8/en_GB.UTF-8/g' /etc/locale.gen
locale-gen

echo "install necessary packages"
apt update
apt install vim htop nvtop git unzip --yes

echo "create user"
groupadd admin
useradd -m -G admin -s $(which bash) $USR

echo "configure bash"
cat >> ~/.bashrc << 'EOF'

export EDITOR=vim
export TERM=xterm-256color
export LANG=en_GB.UTF8

set -o noclobber
alias rm='rm -I'
alias mv='mv -i'
alias cp='cp -i'
EOF

# I'm surprised by how usable the default vim is. But no...
echo "configure vim"
cat >| ~/.vimrc << 'EOF'
set nocompatible
set nomodeline
set encoding=utf8
set showcmd
set number
set scrolloff=5
set foldmethod=syntax
set wildmenu
filetype plugin indent on
syntax on

set termguicolors
" colorscheme slate

set expandtab
set tabstop=4
set smartindent
set softtabstop=4
set smartcase

set omnifunc=syntaxcomplete#Complete
set completeopt=menu,preview
EOF

echo "no auto tmux"
touch ~/.no_auto_tmux
touch $USR_HOME/.no_auto_tmux

echo "transfer basic config"
cp -r -t $USR_HOME .ssh/ .vast_* .bashrc .vimrc
mkdir $WORK_DIR
chown -R $USR: $USR_HOME

echo "configure python related things"
apt install pipx --yes
sudo -H -u $USR bash -c 'curl https://pyenv.run | bash'
cat >> $USR_HOME/.bashrc << 'EOF'

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PATH:/home/kaggle/.local/bin:$PYENV_ROOT/bin"
eval "$(pyenv init -)"
EOF
# somehow the default distribution pipenv is broken.
sudo -H -u $USR bash -c 'pipx install pipenv'
sudo -H -u $USR bash -c 'pipx install tldr'

echo "Now set password for root and $USR."


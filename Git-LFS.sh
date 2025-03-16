# Install git-lfs
# (From https://github.com/git-lfs/git-lfs/wiki/Installation)
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:git-core/ppa
sudo apt-get update
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Add git-lfs to an existing repo
# (From https://github.com/git-lfs/git-lfs/wiki/Tutorial#adding-git-lfs-to-a-pre-existing-repo)
git init .
touch README.md
git add README.md
git commit -m "initial"
git tag one
echo hi > plain.txt
ls > foo.bin
git add plain.txt foo.bin
git commit -m "Add some files"
git tag two
echo bye > plain.txt
ls > bar.bin
ls > foo.bin
git add plain.txt foo.bin bar.bin
git commit -m "Update and add another file"
git tag three
echo bye >> plain.txt
ls > foo.bin
git add plain.txt foo.bin
git commit -m "Update some more"
git tag four

# Now lets decide we want *.bin files to be turned into LFS objects.
git lfs track '*.bin'
git add .gitattributes
git commit -m "Track .bin files"
git tag not_working

# This shows nothing
git lfs ls-files

# Remove the bin files and then add them as LFS objects
git rm --cached *.bin
git add *.bin
git commit -m "Convert last commit to LFS"

# This shows our two tracked .bin files!
git lfs ls-files

# Also this shows that we are LFS; by revealing the object identifier (OID)
git show HEAD:foo.bin

# Shows:
# version https://git-lfs.github.com/spec/v1
# oid sha256:4665a5ea423c2713d436b5ee50593a9640e0018c1550b5a0002f74190d6caea8
# size 36

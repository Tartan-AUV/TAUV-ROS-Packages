sudo apt update && sudo apt install qemu-kvm qemu virt-manager virt-viewer libvirt-clients libvirt-daemon-system bridge-utils virtinst libvirt-daemon qemu binfmt-support qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
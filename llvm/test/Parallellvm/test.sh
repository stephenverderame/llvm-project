c=$(find C -name "*.c")
ir=$(find etc -name "*.cpp")
cpp=$(ls | grep ".cpp$")
echo "$c $ir $cpp" | xargs turnt $1
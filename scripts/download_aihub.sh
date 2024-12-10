#!/bin/bash

echo "=========================================="
echo "aihubshell version 24.01.29 v0.3"
echo "=========================================="

# 상수정의
BASE_URL="https://api.aihub.or.kr"
LOGIN_URL="$BASE_URL/api/loginProcess.do"
BASE_DOWNLOAD_URL="$BASE_URL/down"
MANUAL_URL="$BASE_URL/info/api.do"
BASE_FILETREE_URL="$BASE_URL/info"
DATASET_URL="$BASE_URL/info/dataset.do"

# 사용법 출력 함수
print_usage() {
    manual=$(curl -s "$MANUAL_URL")
    # echo -e "$manual"

    echo $manual | grep -oP '"SJ":"\K[^"]+' | head -n 1
    echo -e '\n'
    # 제목 출력
    echo -e "ENGL_CMGG\t KOREAN_CMGG\t\t\t DETAIL_CN"

    # 데이터 처리 및 출력
    echo "$manual" | awk -F'"' -v RS='},' '
    /ENGL_CMGG/ && /KOREAN_CMGG/ && /DETAIL_CN/ {
        for(i=1; i<=NF; i++) {
            if($i == "ENGL_CMGG") engl=$(i+2)
            if($i == "KOREAN_CMGG") korean=$(i+2)
            if($i == "DETAIL_CN") {
                detail=$(i+2)
                gsub(/\\n/, "\n", detail)
                gsub(/\\t/,"\t\t", detail)
                gsub(/\\/,"", detail)
            }
        }
        printf "%-10s\t %-15s\t|\t %s\n\n", engl, korean, detail
    }'
}

# -help 파라미터가 있는 경우 사용법 출력
if [[ "$1" == "-help" ]]; then
    print_usage
    exit 0
fi

# 파라미터 파싱
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -aihubid) aihubid="$2"; shift ;;
        -aihubpw) aihubpw="$2"; shift ;;
        -mode) 
            mode="$2";
            # 만약 mode가 'l'이고 다음 값이 숫자인 경우 datasetkey로 설정
            if [[ "$mode" == "l" ]]; then
                if [[ "$3" =~ ^[0-9]+$ ]]; then
                    datasetkey="$3";
                    shift;
                fi
            fi
            shift;
            ;;
        -datasetkey) datasetkey="$2"; shift ;;
        -filekey) filekeys="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# mode가 'd'인 경우 datasetkey가 필수값인지 체크
if [[ "$mode" == "d" && -z "$datasetkey" ]]; then
    echo "Error: datasetkey is required when mode is 'd'";
    exit 1;
fi

# 환경 변수에서 아이디와 패스워드 읽어오기 (파라미터가 제공되지 않은 경우에만)
aihubid=${aihubid:-$AIHUB_ID}
aihubpw=${aihubpw:-$AIHUB_PW}

filekeys=${filekeys:-"all"}

# mode에 따른 로직 수행
case $mode in
    d)
        if [[ -z "$aihubid" || -z "$aihubpw" ]]; then
            echo "AIHUB_ID and AIHUB_PW must be provided as parameters or set as environment variables."
            exit 1
        fi
        # 인증
        response_json=$(curl -s -X POST -H "id:$aihubid" -H "pass:$aihubpw" "$LOGIN_URL")
        code=$(echo "$response_json" | sed -n 's/.*"code":\([0-9]*\).*/\1/p')

        # code 값이 200인 경우만 로직 수행
        if [[ "$code" == "200" ]]; then
            echo "Authentication successful."
            
            # 환경 변수 설정
            export AIHUB_ID=$aihubid
            export AIHUB_PW=$aihubpw

            # filekeys 값을 사용하여 요청 전송 및 zip 다운로드
            DOWNLOAD_URL="$BASE_DOWNLOAD_URL/$datasetkey.do"
            
			# 상세한 출력을 화면에 표시하면서 다운로드를 진행합니다.
			http_response=$(curl -L -C - -o "download.tar" -H "id:$aihubid" -H "pass:$aihubpw" -w "\n%{http_code}" "$DOWNLOAD_URL?fileSn=$filekeys")
			http_status=$(echo "$http_response" | tail -n1)
			http_body=$(echo "$http_response" | sed '$d')
			
			# HTTP 상태에 따른 처리를 수행합니다.
			if [ "$http_status" -eq 200 ]; then
			    echo "Request successful with HTTP status $http_status."
			    
			    # curl 명령의 종료 코드를 확인합니다.
			    curl_exit_code=$?
			
			    if [ $curl_exit_code -eq 0 ]; then
			        echo "Download successful."
			        # tar 파일을 해제
                    tar -xvf download.tar

                    echo '잠시 기다려 주세요 병합중 입니다. '
                    merge_parts() {
                        local target_dir="$1"
                        
                        # 해당 폴더에 있는 모든 part 파일들의 prefix를 찾음
                        for prefix in $(ls "$target_dir" | grep '.*\.part[0-9]*$' | sed 's/\(.*\)\.part[0-9]*$/\1/' | sort -u); do
                        	# 파일명에 있는 특수문자 이스케이프 처리
                            escaped_prefix=$(printf '%q' "$prefix")
                            
                            echo "Merging $prefix in $target_dir"
                            
                            # 찾은 prefix로 시작하는 모든 part 파일들을 알파벳순으로 정렬하여 병합
                            #cat "${target_dir}/${prefix}".part* > "${target_dir}/${prefix}"
                            find "${target_dir}" -name "${escaped_prefix}.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "${target_dir}/${prefix}" 
                            
                            # 병합이 완료된 후 part 파일들을 삭제
                            rm "${target_dir}/${prefix}".part*
                        done
                    }

                    # 모든 하위 폴더를 탐색하며 마지막 자식 폴더인 경우에 merge_parts 함수 호출
                    find . -type d | while read -r dir; do
                        if [[ ! -z $(find "$dir" -maxdepth 1 -name '*.part*') ]]; then
                            merge_parts "$dir"
                        fi
                    done

                    echo '병합이 완료 되었습니다. '
                    rm download.tar
			    else
			        echo "Download failed with curl exit code $curl_exit_code."
			    fi
	
			else
			    echo "Download failed with HTTP status $http_status."
			    echo "Error msg:" 
			    cat download.tar
			fi

        else
            echo "Authentication failed. Code: $code"
            exit 1
        fi
        ;;
    l)
        if [[ $datasetkey ]]; then
            # 파일 트리 구조 요청 및 출력
            FILETREE_URL="$BASE_FILETREE_URL/$datasetkey.do"
            echo "Fetching file tree structure..."
            file_tree=$(curl -s "$FILETREE_URL")
            echo "$file_tree"
        else
            # 데이터셋 조회
            echo "Fetching dataset information..."
            dataset_info=$(curl -s "$DATASET_URL")
            echo "$dataset_info"
        fi
        ;;
    *)
        echo "Invalid mode. Please use 'd', 'l' for dataset list, 'l [datasetkey]' for file tree"
        exit 1
        ;;
esac
